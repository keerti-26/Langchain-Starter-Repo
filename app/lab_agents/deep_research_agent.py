"""
Deep Research Agent (Lab 7)
===========================
A ReAct agent that performs web research, summarizes findings, and stores
research context in Supabase `content_context`.

Expected Supabase table schema:

CREATE TABLE content_context (
  id SERIAL PRIMARY KEY,
  url TEXT,
  metadata JSON,
  content TEXT,
  context_vector VECTOR(1536)
)

Required Supabase SQL function for cosine similarity search:

CREATE OR REPLACE FUNCTION match_content_context(
  query_embedding VECTOR(1536),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (id BIGINT, url TEXT, similarity FLOAT)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT cc.id, cc.url,
         1 - (cc.context_vector <=> query_embedding) AS similarity
  FROM content_context cc
  WHERE 1 - (cc.context_vector <=> query_embedding) > match_threshold
  ORDER BY cc.context_vector <=> query_embedding
  LIMIT match_count;
END;
$$;

Required environment variables:
- OPENAI_API_KEY
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY
"""

import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from supabase import Client, create_client

from app.config import get_llm, session_id

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)


def _get_supabase_client() -> Client:
    """Create a Supabase client from environment variables."""
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set to save research context."
        )

    return create_client(supabase_url, supabase_key)


def _get_embeddings_client() -> OpenAIEmbeddings:
    """Embedding client using explicit OpenAI 1536-dim vectors for pgvector compatibility."""
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set to generate embeddings.")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,
        api_key=openai_api_key,
        base_url="https://www.dataexpert.io/api/v1/openai",
        default_headers={"X-Session-ID": session_id},
    )


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return structured search results as JSON.

    Args:
        query: Search query.
        max_results: Number of results to return (1-10).
    """
    max_results = max(1, min(max_results, 10))

    with DDGS() as ddgs:
        rows = list(ddgs.text(query, max_results=max_results))

    normalized = []
    for row in rows:
        normalized.append(
            {
                "title": row.get("title", ""),
                "url": row.get("href", ""),
                "snippet": row.get("body", ""),
            }
        )

    return json.dumps(normalized, indent=2)


@tool
def check_similar_urls_in_supabase(
    content: str, match_threshold: float = 0.85, match_count: int = 5
) -> str:
    """
    Check Supabase for URLs whose stored content is semantically similar to the given
    text using cosine similarity on the context_vector column.

    Use this before saving new content to avoid re-processing URLs already seen.

    Args:
        content: Text to embed and compare against stored content vectors.
        match_threshold: Minimum cosine similarity score (0.0–1.0). Default 0.85.
        match_count: Maximum number of similar results to return. Default 5.
    """
    embeddings = _get_embeddings_client()
    vector = embeddings.embed_query(content)

    if len(vector) != 1536:
        return f"Embedding size mismatch: expected 1536, got {len(vector)}"

    supabase = _get_supabase_client()
    result = supabase.rpc(
        "match_content_context",
        {
            "query_embedding": vector,
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()

    matches = result.data or []
    if not matches:
        return "No similar URLs found above threshold."

    lines = [f"Found {len(matches)} similar URL(s) already in Supabase:"]
    for m in matches:
        lines.append(f"  - id={m['id']} similarity={m['similarity']:.4f} url={m['url']}")
    return "\n".join(lines)


@tool
def save_content_context_to_supabase(url: str, metadata_json: str, content: str) -> str:
    """
    Save research context into Supabase `content_context` with vector embedding.

    Args:
        url: Source URL for this content (or a synthetic URL like research://summary/<topic>)
        metadata_json: JSON string for metadata column.
        content: Full textual content to store and embed.
    """
    metadata = json.loads(metadata_json) if metadata_json else {}

    embeddings = _get_embeddings_client()
    vector = embeddings.embed_query(content)

    if len(vector) != 1536:
        return f"Embedding size mismatch: expected 1536, got {len(vector)}"

    supabase = _get_supabase_client()

    payload = {
        "url": url,
        "metadata": metadata,
        "content": content,
        "context_vector": vector,
    }
    result = supabase.table("content_context").insert(payload).execute()
    inserted = result.data[0] if result.data else {}
    inserted_id = inserted.get("id", "unknown")

    return f"Saved content_context row id={inserted_id} for url={url}"


DEEP_RESEARCH_SYSTEM = textwrap.dedent("""\
    You are a deep research agent.

    Your required workflow for research tasks:
    1) Use `web_search` to gather relevant web sources.
    2) Before saving a URL, call `check_similar_urls_in_supabase` with a snippet of its
       content or title to check if semantically similar content is already stored.
       Skip saving if a match with similarity >= 0.85 already exists.
    3) Synthesize and summarize key findings with citations (URLs).
    4) Persist useful context to Supabase by calling `save_content_context_to_supabase`
       only for URLs that are NOT already seen.
       - Save at least one row per task.
       - metadata_json should include:
         - query
         - saved_at_utc
         - sources (array of URLs)
         - summary_type (e.g., "final_synthesis", "source_note")
    5) From the results you find, use that as inspiration for more web_searches.

    Only stop once you have found at least 100 articles. Make sure to save each one to supabase as you find them!

    Never invent sources. Only cite URLs returned by `web_search`.
""")


def build_deep_research_agent():
    """Create a ReAct deep research agent with web search and Supabase persistence."""
    llm = get_llm(model="gpt-4o", temperature=0)

    @tool
    def utc_now() -> str:
        """Return current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    tools = [web_search, check_similar_urls_in_supabase, save_content_context_to_supabase, utc_now]
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=DEEP_RESEARCH_SYSTEM),
    )
    return agent
