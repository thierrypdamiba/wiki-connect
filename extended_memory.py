"""
Extended Memory System for Wiki Connect

New collections:
- USER_ARTICLES: Generated articles linking two topics
- USER_FEEDBACK: Style preferences for personalization
- LINKUP_CACHE: Dynamic web grounding cache
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
    PayloadSchemaType
)
import cohere


# =============================================================================
# CONFIG
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Use Cohere for embeddings (serverless-friendly)
COHERE_EMBEDDING_MODEL = "embed-multilingual-v2.0"
VECTOR_SIZE = 768

COLLECTIONS = {
    "articles": "user_articles",
    "feedback": "user_feedback",
    "linkup": "linkup_cache",
}


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class UserArticle(BaseModel):
    """Generated article connecting two Wikipedia topics."""
    user_id: str
    article_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    content: str  # Markdown
    topic_a: str  # First Wikipedia page_id
    topic_b: str  # Second Wikipedia page_id
    source_page_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    image_artifact_id: Optional[str] = None
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))


class UserFeedback(BaseModel):
    """User feedback for personalization."""
    user_id: str
    feedback_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    article_id: str  # FK to user_articles
    feedback_type: Literal["text_style", "image_style", "format", "content"] = "text_style"
    feedback_text: str
    rating: int = Field(ge=1, le=5, default=3)
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))


class LinkupResult(BaseModel):
    """Cached web search result from Linkup."""
    result_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str
    url: str
    title: str
    snippet: str
    content: str
    fetched_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    expires_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()) + 86400)  # 24h TTL


# =============================================================================
# CLIENTS
# =============================================================================

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None


def embed(text: str) -> list[float]:
    """Embed text using Cohere API (serverless-friendly)."""
    if not cohere_client:
        raise RuntimeError("COHERE_API_KEY required for embeddings")
    response = cohere_client.embed(
        texts=[text],
        model=COHERE_EMBEDDING_MODEL,
        input_type="search_query"
    )
    return response.embeddings[0]


# =============================================================================
# COLLECTION SETUP
# =============================================================================

def ensure_extended_collections():
    """Create extended collections if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}

    # USER_ARTICLES collection
    if COLLECTIONS["articles"] not in existing:
        client.create_collection(
            collection_name=COLLECTIONS["articles"],
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        for field in ["user_id", "article_id", "topic_a", "topic_b"]:
            client.create_payload_index(
                collection_name=COLLECTIONS["articles"],
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        client.create_payload_index(
            collection_name=COLLECTIONS["articles"],
            field_name="source_page_ids",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # USER_FEEDBACK collection
    if COLLECTIONS["feedback"] not in existing:
        client.create_collection(
            collection_name=COLLECTIONS["feedback"],
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        for field in ["user_id", "feedback_id", "article_id", "feedback_type"]:
            client.create_payload_index(
                collection_name=COLLECTIONS["feedback"],
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    # LINKUP_CACHE collection
    if COLLECTIONS["linkup"] not in existing:
        client.create_collection(
            collection_name=COLLECTIONS["linkup"],
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        for field in ["result_id", "query", "url"]:
            client.create_payload_index(
                collection_name=COLLECTIONS["linkup"],
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    return True


# =============================================================================
# USER_ARTICLES CRUD
# =============================================================================

def store_article(
    user_id: str,
    title: str,
    content: str,
    topic_a: str,
    topic_b: str,
    source_page_ids: list[str] = None,
    source_urls: list[str] = None,
    image_artifact_id: str = None,
) -> UserArticle:
    """Store a generated article."""
    ensure_extended_collections()

    article = UserArticle(
        user_id=user_id,
        title=title,
        content=content,
        topic_a=topic_a.replace(" ", "_"),
        topic_b=topic_b.replace(" ", "_"),
        source_page_ids=source_page_ids or [],
        source_urls=source_urls or [],
        image_artifact_id=image_artifact_id,
    )

    # Embed the title + content for semantic search
    vector = embed(f"{title} {content[:500]}")

    client.upsert(
        collection_name=COLLECTIONS["articles"],
        points=[PointStruct(
            id=article.article_id,
            vector=vector,
            payload=article.model_dump()
        )]
    )

    return article


def search_articles(
    query: str,
    user_id: str = None,
    topic: str = None,
    top_k: int = 5
) -> list[dict]:
    """Search user articles."""
    ensure_extended_collections()
    query_vector = embed(query)

    must_conditions = []
    if user_id:
        must_conditions.append(
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        )
    if topic:
        topic_normalized = topic.replace(" ", "_")
        must_conditions.append(
            FieldCondition(key="source_page_ids", match=MatchAny(any=[topic_normalized]))
        )

    results = client.query_points(
        collection_name=COLLECTIONS["articles"],
        query=query_vector,
        query_filter=Filter(must=must_conditions) if must_conditions else None,
        limit=top_k,
    )

    return [
        {
            "article_id": p.payload.get("article_id"),
            "title": p.payload.get("title"),
            "content": p.payload.get("content"),
            "topic_a": p.payload.get("topic_a"),
            "topic_b": p.payload.get("topic_b"),
            "source_urls": p.payload.get("source_urls", []),
            "image_artifact_id": p.payload.get("image_artifact_id"),
            "score": p.score,
        }
        for p in results.points
    ]


def get_article(article_id: str) -> dict | None:
    """Get a specific article by ID."""
    ensure_extended_collections()
    try:
        points = client.retrieve(
            collection_name=COLLECTIONS["articles"],
            ids=[article_id],
        )
        if points:
            return points[0].payload
    except:
        pass
    return None


# =============================================================================
# USER_FEEDBACK CRUD
# =============================================================================

def store_feedback(
    user_id: str,
    article_id: str,
    feedback_text: str,
    feedback_type: str = "text_style",
    rating: int = 3,
) -> UserFeedback:
    """Store user feedback for personalization."""
    ensure_extended_collections()

    feedback = UserFeedback(
        user_id=user_id,
        article_id=article_id,
        feedback_type=feedback_type,
        feedback_text=feedback_text,
        rating=rating,
    )

    vector = embed(feedback_text)

    client.upsert(
        collection_name=COLLECTIONS["feedback"],
        points=[PointStruct(
            id=feedback.feedback_id,
            vector=vector,
            payload=feedback.model_dump()
        )]
    )

    return feedback


def get_user_preferences(user_id: str, feedback_type: str = None, top_k: int = 10) -> list[dict]:
    """Get user's style preferences from feedback history."""
    ensure_extended_collections()

    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]
    if feedback_type:
        must_conditions.append(
            FieldCondition(key="feedback_type", match=MatchValue(value=feedback_type))
        )

    # Get all feedback for user, sorted by rating
    results = client.scroll(
        collection_name=COLLECTIONS["feedback"],
        scroll_filter=Filter(must=must_conditions),
        limit=top_k,
    )

    return [
        {
            "feedback_type": p.payload.get("feedback_type"),
            "feedback_text": p.payload.get("feedback_text"),
            "rating": p.payload.get("rating"),
            "article_id": p.payload.get("article_id"),
        }
        for p in results[0]
    ]


def search_feedback(query: str, user_id: str, top_k: int = 5) -> list[dict]:
    """Search feedback semantically."""
    ensure_extended_collections()
    query_vector = embed(query)

    results = client.query_points(
        collection_name=COLLECTIONS["feedback"],
        query=query_vector,
        query_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]),
        limit=top_k,
    )

    return [
        {
            "feedback_type": p.payload.get("feedback_type"),
            "feedback_text": p.payload.get("feedback_text"),
            "rating": p.payload.get("rating"),
            "score": p.score,
        }
        for p in results.points
    ]


# =============================================================================
# LINKUP_CACHE CRUD
# =============================================================================

def cache_linkup_result(
    query: str,
    url: str,
    title: str,
    snippet: str,
    content: str,
    ttl_hours: int = 24,
) -> LinkupResult:
    """Cache a Linkup web search result."""
    ensure_extended_collections()

    now = int(datetime.now().timestamp())
    result = LinkupResult(
        query=query,
        url=url,
        title=title,
        snippet=snippet,
        content=content,
        fetched_at=now,
        expires_at=now + (ttl_hours * 3600),
    )

    vector = embed(f"{title} {snippet} {content[:500]}")

    client.upsert(
        collection_name=COLLECTIONS["linkup"],
        points=[PointStruct(
            id=result.result_id,
            vector=vector,
            payload=result.model_dump()
        )]
    )

    return result


def search_linkup_cache(query: str, top_k: int = 5) -> list[dict]:
    """Search cached Linkup results (excludes expired)."""
    ensure_extended_collections()
    query_vector = embed(query)
    now = int(datetime.now().timestamp())

    results = client.query_points(
        collection_name=COLLECTIONS["linkup"],
        query=query_vector,
        limit=top_k,
    )

    # Filter out expired results
    valid_results = []
    for p in results.points:
        expires_at = p.payload.get("expires_at", 0)
        if expires_at > now:
            valid_results.append({
                "url": p.payload.get("url"),
                "title": p.payload.get("title"),
                "snippet": p.payload.get("snippet"),
                "content": p.payload.get("content"),
                "score": p.score,
            })

    return valid_results


def clear_expired_cache():
    """Remove expired Linkup cache entries."""
    ensure_extended_collections()
    now = int(datetime.now().timestamp())

    # Get all expired entries
    results = client.scroll(
        collection_name=COLLECTIONS["linkup"],
        limit=1000,
    )

    expired_ids = [
        p.id for p in results[0]
        if p.payload.get("expires_at", 0) < now
    ]

    if expired_ids:
        client.delete(
            collection_name=COLLECTIONS["linkup"],
            points_selector=expired_ids,
        )

    return len(expired_ids)


# =============================================================================
# UTILITY
# =============================================================================

def get_extended_stats() -> dict:
    """Get counts for extended collections."""
    stats = {}
    for name, collection in COLLECTIONS.items():
        try:
            info = client.get_collection(collection)
            stats[name] = {
                "collection": collection,
                "points": info.points_count,
            }
        except:
            stats[name] = {"collection": collection, "points": 0}
    return stats


def clear_user_extended_data(user_id: str) -> str:
    """Delete all articles and feedback for a user."""
    for collection in [COLLECTIONS["articles"], COLLECTIONS["feedback"]]:
        try:
            client.delete(
                collection_name=collection,
                points_selector=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                ])
            )
        except:
            pass
    return f"Cleared extended data for user {user_id}"
