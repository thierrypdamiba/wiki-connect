"""
FastAPI Server for Wiki Connect

Provides REST API endpoints that the frontend can call.
Wraps the ADK agent functionality.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from google import genai
from google.genai import types
import os

# Initialize Gemini client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

from grounded_memory import search_world, get_stats
from extended_memory import (
    store_article, search_articles, get_article,
    store_feedback, get_user_preferences,
    get_extended_stats, client as qdrant_client, COLLECTIONS,
)
from qdrant_client.models import Filter, FieldCondition, MatchValue
from imagen_client import (
    generate_image_sync,
    generate_image_nano_banana_sync,
    build_connection_image_prompt,
    save_image,
)
from linkup_client import search_and_cache, format_for_grounding, is_available as linkup_available

# Initialize FastAPI
app = FastAPI(
    title="Wiki Connect API",
    description="API for connecting Wikipedia topics and generating articles",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Current user (would be from auth in production)
CURRENT_USER = "default"


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ConnectRequest(BaseModel):
    topic_a: str
    topic_b: str
    user_id: Optional[str] = "default"


class FeedbackRequest(BaseModel):
    article_id: str
    feedback_type: str = "text_style"
    feedback_text: str
    rating: int = 3
    user_id: Optional[str] = "default"


class ImageRequest(BaseModel):
    topic_a: str
    topic_b: str
    connection_summary: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Wiki Connect API",
        "version": "1.0.0",
        "endpoints": [
            "/connect",
            "/search",
            "/generate-image",
            "/feedback",
            "/stats",
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/stats")
async def stats():
    """Get memory statistics."""
    base = get_stats()
    extended = get_extended_stats()
    return {
        "base_collections": base,
        "extended_collections": extended,
        "linkup_available": linkup_available(),
    }


@app.get("/search")
async def search_wikipedia(query: str, top_k: int = 5):
    """Search Wikipedia for grounding."""
    sections = search_world(query, top_k=top_k)

    return {
        "query": query,
        "results": [
            {
                "title": s["page_title"],
                "text": s["section_text"][:500],
                "url": s["url"],
                "score": s["score"],
            }
            for s in sections
        ]
    }


def find_cached_article(topic_a: str, topic_b: str, user_id: str = "default") -> dict | None:
    """Check if we already have an article for this topic combination."""
    try:
        # Normalize topics
        topic_a_norm = topic_a.replace(" ", "_")
        topic_b_norm = topic_b.replace(" ", "_")

        # Search for exact match (either direction)
        results = qdrant_client.scroll(
            collection_name=COLLECTIONS["articles"],
            scroll_filter=Filter(must=[
                FieldCondition(key="topic_a", match=MatchValue(value=topic_a_norm)),
                FieldCondition(key="topic_b", match=MatchValue(value=topic_b_norm)),
            ]),
            limit=1,
        )

        if results[0]:
            return results[0][0].payload

        # Try reverse order
        results = qdrant_client.scroll(
            collection_name=COLLECTIONS["articles"],
            scroll_filter=Filter(must=[
                FieldCondition(key="topic_a", match=MatchValue(value=topic_b_norm)),
                FieldCondition(key="topic_b", match=MatchValue(value=topic_a_norm)),
            ]),
            limit=1,
        )

        if results[0]:
            return results[0][0].payload

    except Exception as e:
        print(f"Cache lookup error: {e}")

    return None


@app.post("/connect")
async def connect_topics(request: ConnectRequest):
    """Connect two topics and generate an article."""
    topic_a = request.topic_a
    topic_b = request.topic_b
    user_id = request.user_id or CURRENT_USER

    # Check cache first
    cached = find_cached_article(topic_a, topic_b, user_id)
    if cached:
        preferences = get_user_preferences(user_id, top_k=10)
        applied_preferences = [
            {"type": p["feedback_type"], "text": p["feedback_text"], "rating": p["rating"]}
            for p in preferences[:5]
        ] if preferences else []

        return {
            "article": {
                "articleId": cached.get("article_id"),
                "title": cached.get("title"),
                "content": cached.get("content"),
                "sources": cached.get("source_urls", []),
            },
            "topicA": topic_a,
            "topicB": topic_b,
            "pageIds": cached.get("source_page_ids", []),
            "appliedPreferences": applied_preferences,
            "cached": True,
        }

    # 1. Search for both topics
    results_a = search_world(topic_a, top_k=4)
    results_b = search_world(topic_b, top_k=4)

    if not results_a:
        raise HTTPException(status_code=404, detail=f"No Wikipedia info for '{topic_a}'")
    if not results_b:
        raise HTTPException(status_code=404, detail=f"No Wikipedia info for '{topic_b}'")

    # 2. Search for connections
    connection_query = f"{topic_a} {topic_b} relationship"
    connection_results = search_world(connection_query, top_k=3)

    # 3. Try Linkup for fresh content if available
    linkup_results = []
    if linkup_available():
        linkup_results = search_and_cache(f"{topic_a} {topic_b} connection", deep=False)

    # 4. Build context for article generation
    context_a = "\n".join([s["section_text"][:300] for s in results_a[:2]])
    context_b = "\n".join([s["section_text"][:300] for s in results_b[:2]])
    context_conn = "\n".join([s["section_text"][:300] for s in connection_results[:2]])

    # 5. Get user preferences from feedback history
    preferences = get_user_preferences(user_id, top_k=10)
    pref_instructions = ""
    if preferences:
        # Group by type for clearer instructions
        text_prefs = [p for p in preferences if p["feedback_type"] == "text_style"]
        format_prefs = [p for p in preferences if p["feedback_type"] == "format"]
        content_prefs = [p for p in preferences if p["feedback_type"] == "content"]

        pref_parts = []
        if text_prefs:
            pref_parts.append(f"Writing style preferences: {'; '.join([p['feedback_text'] for p in text_prefs[:3]])}")
        if format_prefs:
            pref_parts.append(f"Format preferences: {'; '.join([p['feedback_text'] for p in format_prefs[:2]])}")
        if content_prefs:
            pref_parts.append(f"Content preferences: {'; '.join([p['feedback_text'] for p in content_prefs[:2]])}")

        if pref_parts:
            pref_instructions = "\n\n**USER PREFERENCES (apply these to your writing):**\n" + "\n".join(pref_parts)

    # 6. Generate article using Gemini with user preferences
    generation_prompt = f"""You are a Wikipedia-style article writer. Create an engaging article that connects two topics.

**GROUNDING SOURCES (use ONLY these facts):**

About {topic_a}:
{context_a}

About {topic_b}:
{context_b}

Connection context:
{context_conn if context_conn else "Find conceptual bridges between the topics."}
{pref_instructions}

**TASK:**
Write a wiki-style article (800-1200 words) explaining the fascinating connection between "{topic_a}" and "{topic_b}".

Requirements:
1. Use ONLY facts from the grounding sources above
2. Find creative but accurate connections
3. Use clear headers and structure
4. Cite sources inline: "According to Wikipedia's [Topic] article..."
5. Make it engaging and educational
{f"6. IMPORTANT: Apply the user's preferences listed above to customize your writing style" if pref_instructions else ""}

Output the article in Markdown format with a creative title."""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=generation_prompt,
        )
        article_content = response.text

        # Extract title from the generated content (first # header)
        lines = article_content.strip().split("\n")
        title = f"The Connection Between {topic_a} and {topic_b}"
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

    except Exception as e:
        print(f"Gemini generation error: {e}")
        # Fallback to simple template
        title = f"The Connection Between {topic_a} and {topic_b}"
        article_content = f"""# {title}

## Introduction

{topic_a} and {topic_b} might seem unrelated at first glance, but a deeper look reveals fascinating connections.

## About {topic_a}

{context_a}

## About {topic_b}

{context_b}

## The Connection

{context_conn if context_conn else f"The connection between {topic_a} and {topic_b} lies in their shared foundations in human knowledge and discovery."}

## Sources

This article is grounded in Wikipedia's vast knowledge base.
"""

    # 7. Collect sources
    source_urls = list(set(
        [s["url"] for s in results_a] +
        [s["url"] for s in results_b] +
        [s["url"] for s in connection_results]
    ))[:6]

    source_page_ids = list(set(
        [s["page_id"] for s in results_a] +
        [s["page_id"] for s in results_b] +
        [s["page_id"] for s in connection_results]
    ))

    # 8. Store the article
    article = store_article(
        user_id=user_id,
        title=title,
        content=article_content,
        topic_a=topic_a,
        topic_b=topic_b,
        source_page_ids=source_page_ids,
        source_urls=source_urls,
    )

    # Include applied preferences in response
    applied_preferences = []
    if preferences:
        applied_preferences = [
            {"type": p["feedback_type"], "text": p["feedback_text"], "rating": p["rating"]}
            for p in preferences[:5]
        ]

    return {
        "article": {
            "articleId": article.article_id,
            "title": title,
            "content": article_content,
            "sources": source_urls,
        },
        "topicA": topic_a,
        "topicB": topic_b,
        "pageIds": source_page_ids,
        "appliedPreferences": applied_preferences,
    }


@app.post("/connect/stream")
async def connect_topics_stream(request: ConnectRequest):
    """Connect two topics with SSE streaming for real-time updates."""

    async def generate_events() -> AsyncGenerator[str, None]:
        topic_a = request.topic_a
        topic_b = request.topic_b
        user_id = request.user_id or CURRENT_USER

        def send_event(event_type: str, data: dict) -> str:
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"

        def send_decision(decision: str, reasoning: str, action: str) -> str:
            """Send an agent decision event explaining why a choice was made."""
            return send_event("decision", {
                "decision": decision,
                "reasoning": reasoning,
                "action": action,
            })

        # =====================================================================
        # AGENT DECISION: Query Routing Strategy
        # =====================================================================
        yield send_decision(
            decision="Query Routing",
            reasoning=f"Input topics: '{topic_a}' and '{topic_b}'. Need to determine optimal retrieval strategy.",
            action="Route to multi-collection search: (1) Check cache, (2) Wikipedia semantic search, (3) Cross-topic bridging, (4) Optional web fallback"
        )

        # =====================================================================
        # STEP 1: Check Qdrant cache for existing article
        # =====================================================================
        yield send_event("step", {
            "step": "cache",
            "status": "running",
            "message": "Checking Qdrant user_articles collection...",
            "detail": f"Looking for cached article matching topics: {topic_a} ↔ {topic_b}"
        })
        await asyncio.sleep(0.1)

        cached = find_cached_article(topic_a, topic_b, user_id)
        if cached:
            yield send_event("step", {
                "step": "cache",
                "status": "done",
                "message": "Cache HIT! Found existing article",
                "detail": f"Article ID: {cached.get('article_id')[:8]}... | Created: {cached.get('created_at')}",
                "results": [f"Title: {cached.get('title')}", f"Sources: {len(cached.get('source_urls', []))} Wikipedia pages"]
            })

            # Still check for user preferences to show personalization
            yield send_event("step", {
                "step": "preferences",
                "status": "running",
                "message": "Loading user preferences from Qdrant...",
                "detail": f"Querying user_feedback collection for user: {user_id}"
            })
            preferences = get_user_preferences(user_id, top_k=10)
            applied_preferences = [
                {"type": p["feedback_type"], "text": p["feedback_text"], "rating": p["rating"]}
                for p in preferences[:5]
            ] if preferences else []

            yield send_event("step", {
                "step": "preferences",
                "status": "done",
                "message": f"Found {len(preferences)} stored preferences" if preferences else "No preferences found",
                "detail": "Preferences would be applied on regeneration" if preferences else "Give feedback to personalize future articles",
                "results": [f"{p['feedback_type']}: {p['feedback_text']}" for p in preferences[:3]] if preferences else []
            })

            yield send_event("article", {
                "articleId": cached.get("article_id"),
                "title": cached.get("title"),
                "content": cached.get("content"),
                "sources": cached.get("source_urls", []),
                "cached": True,
            })
            yield send_event("complete", {"cached": True, "appliedPreferences": applied_preferences})
            return

        yield send_event("step", {
            "step": "cache",
            "status": "done",
            "message": "Cache MISS - Generating new article",
            "detail": "No existing article found for this topic combination"
        })

        # =====================================================================
        # AGENT DECISION: Memory Retrieval Strategy
        # =====================================================================
        yield send_decision(
            decision="Memory Strategy",
            reasoning="No cached article exists. Must retrieve grounding context from multiple memory layers.",
            action="Execute parallel retrieval: WORLD (Wikipedia 35M) for facts, USER_FEEDBACK for style preferences, optionally WORLD_DYNAMIC (Linkup) for fresh content"
        )

        # =====================================================================
        # STEP 2: Search Qdrant Wikipedia collection for Topic A
        # =====================================================================
        yield send_event("step", {
            "step": "search_a",
            "status": "running",
            "message": f"Searching Qdrant wikipedia_multimodal collection...",
            "detail": f"Query: \"{topic_a}\" | Collection: 35M+ Wikipedia article sections | Using: Cohere embeddings"
        })
        results_a = search_world(topic_a, top_k=4)
        if not results_a:
            yield send_event("error", {"message": f"No Wikipedia info for '{topic_a}'"})
            return
        scores_a = ', '.join([f"{r['score']:.3f}" for r in results_a[:3]])
        yield send_event("step", {
            "step": "search_a",
            "status": "done",
            "message": f"Found {len(results_a)} relevant sections for \"{topic_a}\"",
            "detail": f"Top similarity scores: {scores_a}",
            "results": [f"{r['page_title']}: {r['section_text'][:60]}..." for r in results_a[:3]]
        })

        # =====================================================================
        # STEP 3: Search Qdrant Wikipedia collection for Topic B
        # =====================================================================
        yield send_event("step", {
            "step": "search_b",
            "status": "running",
            "message": f"Searching Qdrant wikipedia_multimodal collection...",
            "detail": f"Query: \"{topic_b}\" | Using same 35M article index"
        })
        results_b = search_world(topic_b, top_k=4)
        if not results_b:
            yield send_event("error", {"message": f"No Wikipedia info for '{topic_b}'"})
            return
        scores_b = ', '.join([f"{r['score']:.3f}" for r in results_b[:3]])
        yield send_event("step", {
            "step": "search_b",
            "status": "done",
            "message": f"Found {len(results_b)} relevant sections for \"{topic_b}\"",
            "detail": f"Top similarity scores: {scores_b}",
            "results": [f"{r['page_title']}: {r['section_text'][:60]}..." for r in results_b[:3]]
        })

        # =====================================================================
        # STEP 4: Search for cross-topic connections
        # =====================================================================
        connection_query = f"{topic_a} {topic_b} relationship"
        yield send_event("step", {
            "step": "connections",
            "status": "running",
            "message": "Finding cross-topic connections in Qdrant...",
            "detail": f"Query: \"{connection_query}\" | Looking for bridging concepts"
        })
        connection_results = search_world(connection_query, top_k=3)
        yield send_event("step", {
            "step": "connections",
            "status": "done",
            "message": f"Found {len(connection_results)} potential connection points",
            "detail": "These sections may contain shared themes or relationships",
            "results": [r["url"] for r in connection_results[:2]]
        })

        # =====================================================================
        # AGENT DECISION: Dynamic Grounding Check
        # =====================================================================
        avg_score = (sum(r['score'] for r in results_a[:2]) + sum(r['score'] for r in results_b[:2])) / 4
        needs_linkup = avg_score < 0.85 or linkup_available()
        yield send_decision(
            decision="Linkup Routing",
            reasoning=f"Wikipedia relevance: avg score {avg_score:.3f}. {'Scores below 0.85 threshold or Linkup available - checking web for supplementary content.' if needs_linkup else 'Wikipedia coverage sufficient, skipping web search.'}",
            action="Query Linkup API for recent web content" if needs_linkup and linkup_available() else "Skip web search, proceed with Wikipedia grounding only"
        )

        # =====================================================================
        # STEP 5: Check Linkup for fresh web content (if available)
        # =====================================================================
        linkup_context = ""
        if linkup_available():
            yield send_event("step", {
                "step": "linkup",
                "status": "running",
                "message": "Searching Linkup for recent web content...",
                "detail": f"Query: \"{topic_a} {topic_b} connection\" | For topics not well covered in Wikipedia"
            })
            try:
                linkup_results = search_and_cache(f"{topic_a} {topic_b} connection", deep=False)
                if linkup_results:
                    linkup_context = format_for_grounding(linkup_results[:2])
                    yield send_event("step", {
                        "step": "linkup",
                        "status": "done",
                        "message": f"Found {len(linkup_results)} fresh web results",
                        "detail": "Results cached in Qdrant linkup_cache collection (24h TTL)",
                        "results": [r.get("title", r.get("url", ""))[:50] for r in linkup_results[:2]]
                    })
                else:
                    yield send_event("step", {
                        "step": "linkup",
                        "status": "done",
                        "message": "No additional web results needed",
                        "detail": "Wikipedia coverage sufficient"
                    })
            except Exception as e:
                yield send_event("step", {
                    "step": "linkup",
                    "status": "done",
                    "message": "Linkup search skipped",
                    "detail": str(e)[:50]
                })
        else:
            yield send_event("step", {
                "step": "linkup",
                "status": "done",
                "message": "Linkup not configured",
                "detail": "Set LINKUP_API_KEY to enable fresh web search fallback"
            })

        # =====================================================================
        # STEP 6: Load user preferences from feedback history
        # =====================================================================
        yield send_event("step", {
            "step": "preferences",
            "status": "running",
            "message": "Querying Qdrant user_feedback collection...",
            "detail": f"User: {user_id} | Looking for writing style preferences"
        })
        preferences = get_user_preferences(user_id, top_k=10)
        pref_instructions = ""
        if preferences:
            text_prefs = [p for p in preferences if p["feedback_type"] == "text_style"]
            format_prefs = [p for p in preferences if p["feedback_type"] == "format"]
            content_prefs = [p for p in preferences if p["feedback_type"] == "content"]

            pref_parts = []
            if text_prefs:
                pref_parts.append(f"Writing style: {'; '.join([p['feedback_text'] for p in text_prefs[:3]])}")
            if format_prefs:
                pref_parts.append(f"Format: {'; '.join([p['feedback_text'] for p in format_prefs[:2]])}")
            if content_prefs:
                pref_parts.append(f"Content: {'; '.join([p['feedback_text'] for p in content_prefs[:2]])}")

            if pref_parts:
                pref_instructions = "\n\n**USER PREFERENCES (apply these to your writing):**\n" + "\n".join(pref_parts)

            yield send_event("step", {
                "step": "preferences",
                "status": "done",
                "message": f"Applying {len(preferences)} user preferences",
                "detail": "These will customize the article's writing style",
                "results": [f"★{'★' * (p['rating']-1)}{'☆' * (5-p['rating'])} {p['feedback_type']}: {p['feedback_text'][:40]}..." for p in preferences[:3]]
            })

            # Decision about how preferences affect output
            yield send_decision(
                decision="Preference Application",
                reasoning=f"Found {len(preferences)} stored preferences. Text style: {len(text_prefs)}, Format: {len(format_prefs)}, Content: {len(content_prefs)}.",
                action=f"Injecting preference instructions into LLM prompt: {pref_parts[0][:50] if pref_parts else 'none'}..."
            )
        else:
            yield send_event("step", {
                "step": "preferences",
                "status": "done",
                "message": "No stored preferences found",
                "detail": "Give feedback on this article to personalize future generations!"
            })

            yield send_decision(
                decision="Preference Application",
                reasoning="No user preferences in USER_FEEDBACK collection. Using default writing style.",
                action="Generate with standard Wikipedia-style formatting. User can provide feedback to personalize future outputs."
            )

        # =====================================================================
        # STEP 7: Build grounding context for LLM
        # =====================================================================
        yield send_event("step", {
            "step": "context",
            "status": "running",
            "message": "Building grounding context from Qdrant + Linkup...",
            "detail": "Combining Wikipedia sections into structured prompt"
        })

        context_a = "\n".join([s["section_text"][:300] for s in results_a[:2]])
        context_b = "\n".join([s["section_text"][:300] for s in results_b[:2]])
        context_conn = "\n".join([s["section_text"][:300] for s in connection_results[:2]])

        total_context_chars = len(context_a) + len(context_b) + len(context_conn)
        yield send_event("step", {
            "step": "context",
            "status": "done",
            "message": f"Context ready: {total_context_chars:,} characters",
            "detail": f"From {len(results_a[:2]) + len(results_b[:2]) + len(connection_results[:2])} Wikipedia sections",
            "results": [
                f"Topic A context: {len(context_a):,} chars",
                f"Topic B context: {len(context_b):,} chars",
                f"Connection context: {len(context_conn):,} chars",
                f"User preferences: {'YES - will apply' if pref_instructions else 'None'}"
            ]
        })

        # =====================================================================
        # STEP 8: Generate article with Gemini streaming
        # =====================================================================
        yield send_event("step", {
            "step": "generate",
            "status": "running",
            "message": "Generating article with Google ADK...",
            "detail": "Grounded generation using Qdrant vectors + Linkup web search"
        })

        generation_prompt = f"""You are a Wikipedia-style article writer. Create an engaging article that connects two topics.

**GROUNDING SOURCES (use ONLY these facts):**

About {topic_a}:
{context_a}

About {topic_b}:
{context_b}

Connection context:
{context_conn if context_conn else "Find conceptual bridges between the topics."}
{pref_instructions}

**TASK:**
Write a wiki-style article (800-1200 words) explaining the fascinating connection between "{topic_a}" and "{topic_b}".

Requirements:
1. Use ONLY facts from the grounding sources above
2. Find creative but accurate connections
3. Use clear headers and structure
4. Cite sources inline: "According to Wikipedia's [Topic] article..."
5. Make it engaging and educational
{f"6. IMPORTANT: Apply the user's preferences listed above to customize your writing style" if pref_instructions else ""}

Output the article in Markdown format with a creative title."""

        try:
            # Use streaming generation - must await to get async generator
            article_content = ""
            stream = await gemini_client.aio.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=generation_prompt,
            )
            async for chunk in stream:
                if chunk.text:
                    article_content += chunk.text
                    yield send_event("content", {"chunk": chunk.text})

            # Extract title
            lines = article_content.strip().split("\n")
            title = f"The Connection Between {topic_a} and {topic_b}"
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            word_count = len(article_content.split())
            yield send_event("step", {
                "step": "generate",
                "status": "done",
                "message": f"Generated {word_count:,} word article",
                "detail": f"Title extracted: \"{title[:50]}...\"" if len(title) > 50 else f"Title: \"{title}\""
            })

        except Exception as e:
            print(f"Gemini streaming error: {e}")
            yield send_event("error", {"message": f"Generation error: {str(e)}"})
            return

        # =====================================================================
        # AGENT DECISION: Memory Storage Strategy
        # =====================================================================
        yield send_decision(
            decision="Memory Storage",
            reasoning=f"Article generated successfully ({word_count} words). Need to persist for future retrieval and caching.",
            action=f"Store in USER_ARTICLES collection with: (1) BGE embedding of title+content for semantic search, (2) topic_a/topic_b fields for exact-match cache lookup, (3) source_page_ids linking back to Wikipedia"
        )

        # =====================================================================
        # STEP 9: Store article in Qdrant
        # =====================================================================
        yield send_event("step", {
            "step": "store",
            "status": "running",
            "message": "Storing article in Qdrant user_articles...",
            "detail": f"Embedding article with BGE model | User: {user_id}"
        })

        source_urls = list(set(
            [s["url"] for s in results_a] +
            [s["url"] for s in results_b] +
            [s["url"] for s in connection_results]
        ))[:6]

        source_page_ids = list(set(
            [s["page_id"] for s in results_a] +
            [s["page_id"] for s in results_b] +
            [s["page_id"] for s in connection_results]
        ))

        article = store_article(
            user_id=user_id,
            title=title,
            content=article_content,
            topic_a=topic_a,
            topic_b=topic_b,
            source_page_ids=source_page_ids,
            source_urls=source_urls,
        )

        yield send_event("step", {
            "step": "store",
            "status": "done",
            "message": f"Article saved! ID: {article.article_id[:8]}...",
            "detail": f"Stored with {len(source_urls)} source URLs | {len(source_page_ids)} page references",
            "results": [
                f"Collection: user_articles",
                f"Topics: {topic_a} ↔ {topic_b}",
                f"Will be cached for future requests"
            ]
        })

        # Final response
        applied_preferences = [
            {"type": p["feedback_type"], "text": p["feedback_text"], "rating": p["rating"]}
            for p in preferences[:5]
        ] if preferences else []

        yield send_event("article", {
            "articleId": article.article_id,
            "title": title,
            "content": article_content,
            "sources": source_urls,
            "cached": False,
        })

        yield send_event("complete", {
            "cached": False,
            "appliedPreferences": applied_preferences,
            "pageIds": source_page_ids,
        })

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    """Generate an illustration for the topic connection using Nano Banana Pro."""
    prompt = build_connection_image_prompt(
        request.topic_a,
        request.topic_b,
        request.connection_summary,
    )

    # Use Nano Banana Pro (Gemini 3 Pro Image) for best quality
    image_bytes, mime_type = generate_image_nano_banana_sync(prompt)

    if not image_bytes:
        # Fallback to Imagen if Nano Banana fails
        image_bytes = generate_image_sync(prompt, model="imagen-fast", aspect_ratio="16:9")
        mime_type = "image/png"

    if not image_bytes:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    # Determine extension from mime type
    ext = "jpg" if "jpeg" in mime_type else "png"

    # Save to disk
    image_id = uuid.uuid4().hex[:8]
    image_dir = Path(__file__).parent / "generated_images"
    image_dir.mkdir(exist_ok=True)
    image_path = image_dir / f"{image_id}.{ext}"

    save_image(image_bytes, str(image_path))

    return {
        "imageId": image_id,
        "imageUrl": f"/images/{image_id}.{ext}",
        "prompt": prompt[:200],
        "model": "nano-banana-pro",
    }


@app.get("/images/{image_filename}")
async def get_image(image_filename: str):
    """Serve a generated image (supports png and jpg)."""
    image_path = Path(__file__).parent / "generated_images" / image_filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine media type from extension
    media_type = "image/jpeg" if image_filename.endswith(".jpg") else "image/png"

    return FileResponse(image_path, media_type=media_type)


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on an article."""
    user_id = request.user_id or CURRENT_USER

    feedback = store_feedback(
        user_id=user_id,
        article_id=request.article_id,
        feedback_text=request.feedback_text,
        feedback_type=request.feedback_type,
        rating=request.rating,
    )

    return {
        "feedbackId": feedback.feedback_id,
        "message": "Feedback saved successfully",
    }


@app.get("/preferences/{user_id}")
async def get_preferences(user_id: str, feedback_type: Optional[str] = None):
    """Get user preferences."""
    prefs = get_user_preferences(user_id, feedback_type=feedback_type, top_k=10)
    return {
        "userId": user_id,
        "preferences": prefs,
    }


@app.get("/articles")
async def list_articles(query: str = "", user_id: str = "default", top_k: int = 10):
    """List user's articles."""
    articles = search_articles(query or "connection", user_id=user_id, top_k=top_k)
    return {
        "articles": articles,
        "count": len(articles),
    }


@app.get("/articles/{article_id}")
async def get_article_by_id(article_id: str):
    """Get a specific article."""
    article = get_article(article_id)

    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return article


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
