"""
Wikipedia-Grounded Memory System

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│  WORLD (Wikipedia)  - Canonical source of truth (35M sections)          │
│    page_id, page_title, section_id, section_title, section_text, url    │
├─────────────────────────────────────────────────────────────────────────┤
│  USER_NOTES        - Per-user overlays keyed to Wikipedia pages         │
│    user_id, page_id, note_type, note_text, tags                         │
├─────────────────────────────────────────────────────────────────────────┤
│  USER_HISTORY      - Episodic memory with related_page_ids              │
│    user_id, session_id, text, related_page_ids, timestamp               │
└─────────────────────────────────────────────────────────────────────────┘

Key principle: Every memory item either points to a Wikipedia page or is
used to *interpret* a Wikipedia page. All factual claims must trace back
to Wikipedia sections.

API:
- search_world(query, top_k) → Wikipedia sections
- search_notes(query, user_id, page_ids, top_k) → user overlays for pages
- search_history(query, user_id, page_ids, top_k) → past conversations
- store_note(user_id, page_id, note_type, note_text, tags) → save overlay
- store_history(user_id, session_id, text, page_ids) → log conversation
- grounded_answer(query, user_id) → full retrieval pipeline
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
    SearchParams, PayloadSchemaType
)
# =============================================================================
# CONFIG
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Use Cohere for embeddings (serverless-friendly, no local model download)
VECTOR_SIZE = 768  # Cohere embed-multilingual-v2.0 size

# Use full Wikipedia if Cohere key available, otherwise use our demo collection
USE_FULL_WIKIPEDIA = bool(COHERE_API_KEY)

COLLECTIONS = {
    "world": "wikipedia_multimodal" if USE_FULL_WIKIPEDIA else "wikipedia_demo",
    "notes": "user_notes",
    "history": "user_history",
}


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class UserNote(BaseModel):
    """Per-user overlay keyed to Wikipedia pages."""
    user_id: str
    page_id: str  # FK into world_wikipedia
    section_id: Optional[str] = None  # Optional, more precise
    note_type: Literal["preference", "summary", "example", "warning", "question"] = "preference"
    note_text: str
    tags: list[str] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))


class UserHistory(BaseModel):
    """Episodic memory with links to Wikipedia pages."""
    user_id: str
    session_id: str
    turn_index: int = 0
    role: Literal["user", "assistant"] = "user"
    text: str
    related_page_ids: list[str] = Field(default_factory=list)  # Pages involved
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()))


class GroundedContext(BaseModel):
    """Complete context for grounded answer generation."""
    query: str
    user_id: str
    wiki_sections: list[dict]  # From world collection
    user_notes: list[dict]     # Overlays for these pages
    user_history: list[dict]   # Past conversations about these pages
    page_ids: list[str]        # Wikipedia pages being referenced


# =============================================================================
# CLIENTS
# =============================================================================

# Qdrant client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

# Cohere client for all embeddings (serverless-friendly)
import cohere
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
COHERE_EMBEDDING_MODEL = "embed-multilingual-v2.0"


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

def ensure_collections():
    """Create user collections if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}

    # User Notes collection
    if COLLECTIONS["notes"] not in existing:
        client.create_collection(
            collection_name=COLLECTIONS["notes"],
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        for field in ["user_id", "page_id", "section_id", "note_type"]:
            client.create_payload_index(
                collection_name=COLLECTIONS["notes"],
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    # User History collection
    if COLLECTIONS["history"] not in existing:
        client.create_collection(
            collection_name=COLLECTIONS["history"],
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        for field in ["user_id", "session_id", "role"]:
            client.create_payload_index(
                collection_name=COLLECTIONS["history"],
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        # Array index for related_page_ids
        client.create_payload_index(
            collection_name=COLLECTIONS["history"],
            field_name="related_page_ids",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    return True


# =============================================================================
# SEARCH: WORLD (Wikipedia)
# =============================================================================

def search_world(query: str, top_k: int = 8) -> list[dict]:
    """
    Search Wikipedia for grounding context.
    Uses Cloud Inference with Cohere if API key available, otherwise local BGE.
    Returns sections with page_id, title, text, url.
    """
    from qdrant_client import models

    if USE_FULL_WIKIPEDIA and cohere_client:
        # Use Cohere SDK to embed, then search
        response = cohere_client.embed(
            texts=[query],
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_query",
        )
        query_vector = response.embeddings[0]
        results = client.query_points(
            collection_name=COLLECTIONS["world"],
            query=query_vector,
            using="text",
            limit=top_k,
            search_params=SearchParams(indexed_only=True),
        )
    else:
        # Use local BGE embeddings for our demo collection
        query_vector = embed(query)
        results = client.query_points(
            collection_name=COLLECTIONS["world"],
            query=query_vector,
            limit=top_k,
        )

    sections = []
    for point in results.points:
        payload = point.payload
        sections.append({
            "page_id": payload.get("title", "").replace(" ", "_"),  # Normalize
            "page_title": payload.get("title", "Unknown"),
            "section_title": "",
            "section_text": payload.get("text", ""),
            "url": payload.get("url", f"https://en.wikipedia.org/wiki/{payload.get('title', '')}"),
            "score": point.score,
        })

    return sections


# =============================================================================
# SEARCH: USER NOTES (overlays for Wikipedia pages)
# =============================================================================

def search_notes(
    query: str,
    user_id: str,
    page_ids: list[str] = None,
    top_k: int = 8
) -> list[dict]:
    """
    Search user notes, optionally filtered to specific Wikipedia pages.
    """
    ensure_collections()
    query_vector = embed(query)

    # Build filter
    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]
    if page_ids:
        must_conditions.append(
            FieldCondition(key="page_id", match=MatchAny(any=page_ids))
        )

    results = client.query_points(
        collection_name=COLLECTIONS["notes"],
        query=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=top_k,
    )

    return [
        {
            "page_id": p.payload.get("page_id"),
            "note_type": p.payload.get("note_type"),
            "note_text": p.payload.get("note_text"),
            "tags": p.payload.get("tags", []),
            "score": p.score,
        }
        for p in results.points
    ]


# =============================================================================
# SEARCH: USER HISTORY (episodic with page links)
# =============================================================================

def search_history(
    query: str,
    user_id: str,
    page_ids: list[str] = None,
    top_k: int = 8
) -> list[dict]:
    """
    Search user conversation history, optionally filtered to pages.
    """
    ensure_collections()
    query_vector = embed(query)

    # Build filter
    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]
    if page_ids:
        # Match history entries that reference any of these pages
        must_conditions.append(
            FieldCondition(key="related_page_ids", match=MatchAny(any=page_ids))
        )

    results = client.query_points(
        collection_name=COLLECTIONS["history"],
        query=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=top_k,
    )

    return [
        {
            "text": p.payload.get("text"),
            "role": p.payload.get("role"),
            "related_page_ids": p.payload.get("related_page_ids", []),
            "session_id": p.payload.get("session_id"),
            "score": p.score,
        }
        for p in results.points
    ]


# =============================================================================
# STORE: USER NOTES
# =============================================================================

def store_note(
    user_id: str,
    page_id: str,
    note_text: str,
    note_type: str = "preference",
    section_id: str = None,
    tags: list[str] = None
) -> str:
    """Store a user note linked to a Wikipedia page."""
    ensure_collections()

    note = UserNote(
        user_id=user_id,
        page_id=page_id,
        section_id=section_id,
        note_type=note_type,
        note_text=note_text,
        tags=tags or [],
    )

    vector = embed(note_text)
    point_id = uuid.uuid4().hex

    client.upsert(
        collection_name=COLLECTIONS["notes"],
        points=[PointStruct(id=point_id, vector=vector, payload=note.model_dump())]
    )

    return f"Stored note for {page_id}: {note_text[:50]}..."


# =============================================================================
# STORE: USER HISTORY
# =============================================================================

def store_history(
    user_id: str,
    session_id: str,
    text: str,
    related_page_ids: list[str] = None,
    role: str = "user",
    turn_index: int = 0
) -> str:
    """Store a conversation turn linked to Wikipedia pages."""
    ensure_collections()

    history = UserHistory(
        user_id=user_id,
        session_id=session_id,
        turn_index=turn_index,
        role=role,
        text=text,
        related_page_ids=related_page_ids or [],
    )

    vector = embed(text)
    point_id = uuid.uuid4().hex

    client.upsert(
        collection_name=COLLECTIONS["history"],
        points=[PointStruct(id=point_id, vector=vector, payload=history.model_dump())]
    )

    return f"Stored history: {text[:50]}..."


# =============================================================================
# GROUNDED ANSWER PIPELINE
# =============================================================================

def grounded_answer(query: str, user_id: str, top_k: int = 8) -> GroundedContext:
    """
    Full retrieval pipeline for grounded answer generation.

    Steps:
    1. Search Wikipedia for grounding context
    2. Extract page_ids from results
    3. Pull user notes for those pages
    4. Pull user history for those pages
    5. Return complete context

    The LLM should:
    - Use wiki_sections for factual claims (cite them!)
    - Use user_notes to personalize tone/examples
    - Use user_history for continuity
    """
    ensure_collections()

    # Step 1: Search Wikipedia first (grounding)
    wiki_sections = search_world(query, top_k=top_k)

    # Step 2: Extract unique page_ids
    page_ids = list(set(s["page_id"] for s in wiki_sections if s.get("page_id")))

    # Step 3: Pull user overlays for these pages
    user_notes = []
    if page_ids:
        user_notes = search_notes(query, user_id, page_ids=page_ids, top_k=top_k)

    # Step 4: Pull user history for these pages
    user_history = []
    if page_ids:
        user_history = search_history(query, user_id, page_ids=page_ids, top_k=top_k)

    return GroundedContext(
        query=query,
        user_id=user_id,
        wiki_sections=wiki_sections,
        user_notes=user_notes,
        user_history=user_history,
        page_ids=page_ids,
    )


# =============================================================================
# UTILITY
# =============================================================================

def get_stats() -> dict:
    """Get counts for all collections."""
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


def clear_user_data(user_id: str) -> str:
    """Delete all notes and history for a user."""
    for collection in [COLLECTIONS["notes"], COLLECTIONS["history"]]:
        try:
            client.delete(
                collection_name=collection,
                points_selector=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                ])
            )
        except:
            pass
    return f"Cleared all data for user {user_id}"
