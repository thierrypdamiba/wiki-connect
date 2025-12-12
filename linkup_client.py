"""
Linkup Client for Dynamic Web Grounding

Uses Linkup API to fetch real-time web content when Wikipedia
is stale or missing information.

Requires LINKUP_API_KEY environment variable.
"""

import os
from datetime import datetime, date
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from linkup import LinkupClient

# Initialize client
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
client = LinkupClient(api_key=LINKUP_API_KEY) if LINKUP_API_KEY else None


def search_web(
    query: str,
    depth: Literal["standard", "deep"] = "standard",
    output_type: Literal["searchResults", "sourcedAnswer", "structured"] = "searchResults",
    from_date: date = None,
    to_date: date = None,
) -> dict:
    """
    Search the web using Linkup API.

    Args:
        query: Search query
        depth: "standard" for fast simple queries, "deep" for complex agentic search
        output_type: "searchResults" for documents, "sourcedAnswer" for answer with sources
        from_date: Filter results published after this date
        to_date: Filter results published before this date

    Returns:
        Search results dict with 'results' or 'answer' depending on output_type
    """
    if not client:
        return {"error": "LINKUP_API_KEY not set", "results": []}

    try:
        response = client.search(
            query=query,
            depth=depth,
            output_type=output_type,
            from_date=from_date,
            to_date=to_date,
        )

        if output_type == "searchResults":
            return {
                "query": query,
                "results": [
                    {
                        "url": r.url,
                        "title": r.name,
                        "snippet": r.content[:500] if r.content else "",
                        "content": r.content or "",
                    }
                    for r in response.results
                ] if hasattr(response, 'results') else []
            }
        elif output_type == "sourcedAnswer":
            return {
                "query": query,
                "answer": response.answer if hasattr(response, 'answer') else "",
                "sources": [
                    {"url": s.url, "title": s.name}
                    for s in response.sources
                ] if hasattr(response, 'sources') else []
            }
        else:
            return {"query": query, "response": response}

    except Exception as e:
        return {"error": str(e), "results": []}


async def search_web_async(
    query: str,
    depth: Literal["standard", "deep"] = "standard",
    output_type: Literal["searchResults", "sourcedAnswer"] = "searchResults",
    from_date: date = None,
    to_date: date = None,
) -> dict:
    """Async version of search_web."""
    if not client:
        return {"error": "LINKUP_API_KEY not set", "results": []}

    try:
        response = await client.async_search(
            query=query,
            depth=depth,
            output_type=output_type,
            from_date=from_date,
            to_date=to_date,
        )

        if output_type == "searchResults":
            return {
                "query": query,
                "results": [
                    {
                        "url": r.url,
                        "title": r.name,
                        "snippet": r.content[:500] if r.content else "",
                        "content": r.content or "",
                    }
                    for r in response.results
                ] if hasattr(response, 'results') else []
            }
        else:
            return {
                "query": query,
                "answer": response.answer if hasattr(response, 'answer') else "",
                "sources": [
                    {"url": s.url, "title": s.name}
                    for s in response.sources
                ] if hasattr(response, 'sources') else []
            }

    except Exception as e:
        return {"error": str(e), "results": []}


def get_sourced_answer(query: str, deep: bool = False) -> dict:
    """
    Get a concise answer with sources - useful for grounding.

    Args:
        query: Question to answer
        deep: Use deep agentic search for complex queries

    Returns:
        Dict with 'answer' and 'sources'
    """
    return search_web(
        query=query,
        depth="deep" if deep else "standard",
        output_type="sourcedAnswer",
    )


def search_recent(query: str, days_back: int = 7) -> dict:
    """
    Search for recent content only.

    Args:
        query: Search query
        days_back: How many days back to search

    Returns:
        Search results from the last N days
    """
    from_date = date.today().replace(day=date.today().day - days_back)
    return search_web(query=query, from_date=from_date)


# =============================================================================
# INTEGRATION WITH EXTENDED MEMORY
# =============================================================================

def search_and_cache(query: str, deep: bool = False) -> list[dict]:
    """
    Search web and cache results in Qdrant.

    Args:
        query: Search query
        deep: Use deep search for complex queries

    Returns:
        List of search results (also cached)
    """
    from extended_memory import cache_linkup_result, search_linkup_cache

    # Check cache first
    cached = search_linkup_cache(query, top_k=3)
    if cached and len(cached) >= 2:
        return cached

    # Fetch fresh results
    results = search_web(
        query=query,
        depth="deep" if deep else "standard",
        output_type="searchResults",
    )

    if "error" in results:
        return cached if cached else []

    # Cache the results
    for r in results.get("results", [])[:5]:  # Cache top 5
        cache_linkup_result(
            query=query,
            url=r.get("url", ""),
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            content=r.get("content", ""),
            ttl_hours=24,
        )

    return results.get("results", [])


def is_available() -> bool:
    """Check if Linkup is configured and available."""
    return client is not None


def format_for_grounding(results: list[dict]) -> str:
    """
    Format Linkup results for LLM grounding context.

    Args:
        results: List of search results

    Returns:
        Formatted string for LLM context
    """
    if not results:
        return "No web results found."

    output = ["**Web Search Results (Linkup):**\n"]
    for i, r in enumerate(results[:5], 1):
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        snippet = r.get("snippet", r.get("content", ""))[:300]
        output.append(f"{i}. **{title}**")
        output.append(f"   {snippet}...")
        output.append(f"   Source: {url}\n")

    return "\n".join(output)
