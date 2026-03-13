"""
live_search.py
───────────────
REgalys Live Literature Augmentation

Augments REgalys query results with fresh literature from Semantic Scholar
at query time — covering papers published AFTER the knowledge base was frozen.

Why this matters:
  Your KB was built through early 2025. TTE, CCW, competing events, and
  gabapentinoid literature continues to grow weekly. This module searches
  Semantic Scholar live, converts abstracts into the same chunk format
  used by HybridRetriever, and merges them into the retrieval results
  before synthesis. Claude sees both your curated KB AND the latest evidence.

Architecture:
  User query
    → [existing KB retrieval] ────────────────────────────────┐
    → [Semantic Scholar live search] → normalise → re-score   │
                                                               ↓
                                          merge + deduplicate by PMID
                                                               ↓
                                      Cohere rerank (combined pool)
                                                               ↓
                                             Claude synthesis

Integration:
  from live_search import LiveSearchAugmenter
  augmenter = LiveSearchAugmenter()

  # Augment existing chunks with fresh S2 results
  augmented_chunks = augmenter.augment(query, existing_chunks, top_n_fresh=5)
"""

import time
import requests
from typing import Optional
from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Scholar API
# ─────────────────────────────────────────────────────────────────────────────

_S2_BASE   = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = ",".join([
    "paperId", "externalIds", "title", "abstract",
    "authors", "year", "venue", "openAccessPdf", "citationCount",
])


def _s2_headers() -> dict:
    headers = {"Accept": "application/json"}
    if cfg.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = cfg.SEMANTIC_SCHOLAR_API_KEY
    return headers


def _s2_search(query: str, limit: int = 20, year_from: int = 2020) -> list[dict]:
    """
    Searches Semantic Scholar for papers matching a query.

    Args:
        query:     natural language search string
        limit:     max results (1-100)
        year_from: only return papers published >= this year
                   Set to 2025 for truly fresh-only results,
                   or 2020 for broader recent coverage.

    Returns:
        list of raw S2 paper dicts
    """
    params = {
        "query":  query,
        "limit":  min(limit, 100),
        "fields": _S2_FIELDS,
        "year":   f"{year_from}-",   # S2 supports year range filter
    }

    try:
        resp = requests.get(
            f"{_S2_BASE}/paper/search",
            headers = _s2_headers(),
            params  = params,
            timeout = 15,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"  [LiveSearch] S2 search failed: {e}")
        return []


def _s2_paper_to_chunk(paper: dict) -> Optional[dict]:
    """
    Converts a Semantic Scholar paper dict into the same chunk format
    used by HybridRetriever — so it can be seamlessly merged and reranked
    with KB chunks.

    Returns None if the paper has no usable abstract.
    """
    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return None

    ext_ids = paper.get("externalIds") or {}
    pmid    = str(ext_ids.get("PubMed", ""))
    doi     = ext_ids.get("DOI", "")

    authors_list = paper.get("authors") or []
    author_names = [a.get("name", "") for a in authors_list[:6] if a.get("name")]
    author_str   = ", ".join(author_names)
    if len(authors_list) > 6:
        author_str += " et al."

    year    = str(paper.get("year") or "")
    journal = paper.get("venue")    or ""
    title   = (paper.get("title")   or "").strip()

    # Build citation string matching your KB convention
    citation = f"{author_str} ({year}). {title}. {journal}."
    if doi:
        citation += f" https://doi.org/{doi}"

    # Unique ID for this chunk — S2 paper ID prefixed so we can identify source
    chunk_id = f"S2_{paper.get('paperId', doi or title[:40])}"

    return {
        # Core chunk fields (matching HybridRetriever output format)
        "chunk_id":      chunk_id,
        "text":          abstract,
        "text_original": abstract,
        "title":         title,
        "authors":       author_str,
        "year":          year,
        "journal":       journal,
        "pmid":          pmid,
        "doi":           doi,
        "citation":      citation,
        "section":       "abstract",

        # Live search metadata
        "source":           "semantic_scholar_live",
        "citation_count":   paper.get("citationCount", 0),
        "retrieval_score":  0.0,   # will be set after reranking
        "is_live_result":   True,  # flag for UI display
    }


# ─────────────────────────────────────────────────────────────────────────────
# Augmenter class
# ─────────────────────────────────────────────────────────────────────────────

class LiveSearchAugmenter:
    """
    Augments REgalys KB retrieval with live Semantic Scholar results.

    The augmenter runs a fast S2 search at query time, converts results
    to the standard chunk format, deduplicates against existing chunks
    (by PMID), and returns a merged pool for reranking.

    Usage:
        augmenter = LiveSearchAugmenter()

        # Returns merged chunk list: KB results + fresh S2 results
        merged = augmenter.augment(
            query          = user_query,
            existing_chunks = kb_chunks,
            top_n_fresh    = 5,    # how many fresh results to add
            year_from      = 2024, # only papers from this year onward
        )

    Cost: zero — Semantic Scholar API is free with your key.
    Latency: ~1-2 seconds for the S2 search call.
    """

    def __init__(self):
        self._available = bool(cfg.SEMANTIC_SCHOLAR_API_KEY)
        if not self._available:
            print("  [LiveSearch] SEMANTIC_SCHOLAR_API_KEY not set — live augmentation disabled")

    def augment(
        self,
        query:           str,
        existing_chunks: list[dict],
        top_n_fresh:     int = 5,
        year_from:       int = 2024,
    ) -> list[dict]:
        """
        Augments existing KB chunks with fresh S2 results.

        Deduplication: S2 results with a PMID already in existing_chunks
        are silently dropped — no duplicates enter the synthesis context.

        Args:
            query:           original user query
            existing_chunks: chunks already retrieved from the KB
            top_n_fresh:     maximum fresh S2 papers to add (default 5)
            year_from:       only include papers from this year onward

        Returns:
            merged list — existing chunks first, then appended fresh chunks
            (ordering preserved for downstream reranking)
        """
        if not self._available:
            return existing_chunks

        # Build set of PMIDs already in existing chunks (for deduplication)
        existing_pmids = {
            str(c.get("pmid", ""))
            for c in existing_chunks
            if c.get("pmid")
        }

        # Search S2
        raw_papers = _s2_search(query, limit=top_n_fresh * 3, year_from=year_from)

        fresh_chunks = []
        for paper in raw_papers:
            chunk = _s2_paper_to_chunk(paper)
            if chunk is None:
                continue

            # Deduplicate by PMID
            pmid = chunk.get("pmid", "")
            if pmid and pmid in existing_pmids:
                continue

            fresh_chunks.append(chunk)
            if pmid:
                existing_pmids.add(pmid)

            if len(fresh_chunks) >= top_n_fresh:
                break

        if fresh_chunks:
            print(f"  [LiveSearch] Added {len(fresh_chunks)} fresh S2 results (>= {year_from})")
        else:
            print(f"  [LiveSearch] No new results found from S2 for this query")

        return existing_chunks + fresh_chunks

    @property
    def is_available(self) -> bool:
        return self._available
