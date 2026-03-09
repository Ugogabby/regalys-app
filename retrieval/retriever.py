"""
retrieval/retriever.py
───────────────────────
Hybrid retriever: BM25 + semantic search + Reciprocal Rank Fusion (RRF).

Why hybrid retrieval?
  BM25 (keyword search) is excellent at finding exact terms:
    "immortal time bias", "Fine-Gray subdistribution hazard", "T40.2"
  Semantic search (vector similarity) is excellent at finding concepts:
    "how do you handle competing events" finds chunks about Fine-Gray
    even if they don't use those exact words.
  Combining both with RRF gives you the best of both worlds.

Pipeline:
  Query → [BM25 top-K] + [Voyage embedding → Pinecone top-K]
        → RRF fusion → merged ranked list
        → Cohere reranker (cross-encoder, optional)
        → top-N chunks returned to app

Reciprocal Rank Fusion (RRF):
  For each chunk, its RRF score = sum of 1/(rank + k) across all lists
  where k=60 is a smoothing constant. A chunk ranked #1 in both lists
  scores much higher than one ranked #1 in only one list.
  This rewards consistent relevance across retrieval methods.

Cloud mode vs local mode:
  Local:  all_chunks.json loaded into RAM → BM25 + semantic + RRF fusion
  Cloud:  all_chunks.json not available → semantic search only via Pinecone
          chunk text is reconstructed from Pinecone metadata cache
          BM25 is silently disabled — quality remains excellent
"""

import json
from pathlib import Path
from typing import Optional

import voyageai
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from config import cfg


# ── RRF smoothing constant ────────────────────────────────────────────────────
# k=60 is the standard value from the original RRF paper (Cormack et al. 2009)
RRF_K = 60


class HybridRetriever:
    """
    Hybrid retriever combining BM25, semantic search, and optional reranking.

    Local mode (all_chunks.json present):
      - Loads all chunks into RAM
      - Builds BM25 index
      - Runs BM25 + semantic + RRF fusion

    Cloud mode (all_chunks.json absent):
      - BM25 disabled
      - Semantic search only via Pinecone
      - Chunk text reconstructed from Pinecone metadata cache
    """

    def __init__(self):
        self.chunks               = []   # all chunks (local mode only)
        self.chunk_lookup         = {}   # chunk_id → chunk dict (local mode only)
        self.bm25                 = None # BM25Okapi index (local mode only)
        self.voyage               = None # Voyage AI client
        self.pinecone_idx         = None # Pinecone index object
        self._pinecone_meta_cache = {}   # chunk_id → metadata (cloud mode)
        self._cloud_mode          = False

        self._load_chunks()
        self._build_bm25()
        self._connect_voyage()
        self._connect_pinecone()

    def _load_chunks(self):
        """
        Loads all_chunks.json into memory for local mode.
        Gracefully falls back to cloud mode if file is not present.
        """
        chunks_path = Path(cfg.CHUNKS_DIR) / "all_chunks.json"

        if not chunks_path.exists():
            # Cloud mode — no local chunks file, semantic search only
            self._cloud_mode  = True
            self.chunks       = []
            self.chunk_lookup = {}
            print("  Cloud mode: all_chunks.json not found — BM25 disabled, using semantic search only")
            return

        print("Loading chunks into memory...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.chunk_lookup = {c["chunk_id"]: c for c in self.chunks}
        self._cloud_mode  = False
        print(f"  ✓ Loaded {len(self.chunks):,} chunks (local mode)")

    def _build_bm25(self):
        """
        Builds BM25 index from chunk texts.
        Skipped automatically in cloud mode.
        """
        if self._cloud_mode or not self.chunks:
            self.bm25 = None
            print("  BM25 index: skipped (cloud mode)")
            return

        print("Building BM25 index...")
        corpus = [
            c.get("text_original", c["text"]).lower().split()
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus)
        print(f"  ✓ BM25 index built over {len(corpus):,} documents")

    def _connect_voyage(self):
        """Initializes the Voyage AI client for query embedding."""
        self.voyage = voyageai.Client(api_key=cfg.VOYAGE_API_KEY)
        print("  ✓ Voyage AI connected")

    def _connect_pinecone(self):
        """Connects to the Pinecone index."""
        pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
        self.pinecone_idx = pc.Index(cfg.PINECONE_INDEX)
        print("  ✓ Pinecone connected")

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Runs BM25 keyword search.
        Returns empty list in cloud mode (BM25 index not available).
        """
        if self.bm25 is None:
            return []

        tokenized = query.lower().split()
        scores    = self.bm25.get_scores(tokenized)

        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self.chunks[idx]["chunk_id"]
                results.append((chunk_id, float(scores[idx])))

        return results

    def _semantic_search(self, query: str, top_k: int,
                         filters: Optional[dict] = None) -> list[tuple[str, float]]:
        """
        Runs semantic search via Voyage AI embedding + Pinecone ANN.

        In cloud mode, caches Pinecone metadata for chunk text reconstruction.

        Args:
            query:   search query string
            top_k:   number of results to return
            filters: optional Pinecone metadata filters
                     e.g. {"section": {"$in": ["methods", "results"]}}

        Returns:
            list of (chunk_id, cosine_score) tuples
        """
        # Embed the query
        # input_type="query" optimised for search (vs "document" for storage)
        result    = self.voyage.embed(
            [query],
            model      = "voyage-3",
            input_type = "query",
        )
        query_vec = result.embeddings[0]

        # Query Pinecone
        kwargs = {
            "vector":           query_vec,
            "top_k":            top_k,
            "include_metadata": True,
        }
        if filters:
            kwargs["filter"] = filters

        response = self.pinecone_idx.query(**kwargs)

        results = []
        for match in response.matches:
            # Cache metadata for cloud mode chunk assembly
            # Pinecone metadata contains authors, year, journal, section, etc.
            meta = dict(match.metadata)
            meta["text_original"] = meta.get("text", "")
            self._pinecone_meta_cache[match.id] = meta
            results.append((match.id, float(match.score)))

        return results

    def _rrf_fusion(self, *ranked_lists: list[tuple[str, float]],
                    k: int = RRF_K) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion across multiple ranked lists.

        Formula: score(d) = Σ 1 / (k + rank(d, list_i))
        where rank is 1-indexed and k=60 is the smoothing constant.

        Args:
            *ranked_lists: any number of (chunk_id, score) lists
            k:             RRF smoothing constant

        Returns:
            fused list of (chunk_id, rrf_score) sorted descending
        """
        rrf_scores = {}

        for ranked_list in ranked_lists:
            for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                rrf_scores[chunk_id] += 1.0 / (k + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return fused

    def retrieve(
        self,
        query:        str,
        top_k:        int            = None,
        filters:      Optional[dict] = None,
        use_bm25:     bool           = True,
        use_semantic: bool           = True,
    ) -> list[dict]:
        """
        Main retrieval method — runs hybrid search and returns ranked chunks.

        In local mode:  BM25 + semantic + RRF fusion
        In cloud mode:  semantic only (BM25 silently skipped)

        Args:
            query:        search query (natural language or keyword)
            top_k:        number of chunks to return (default from config)
            filters:      optional Pinecone metadata filters
            use_bm25:     include BM25 results (ignored in cloud mode)
            use_semantic: include semantic search results

        Returns:
            list of chunk dicts with added 'retrieval_score' field,
            sorted by relevance descending
        """
        if top_k is None:
            top_k = cfg.TOP_K

        candidate_k  = top_k * 3
        ranked_lists = []

        # ── BM25 search (local mode only) ─────────────────────────────────────
        if use_bm25 and not self._cloud_mode:
            bm25_results = self._bm25_search(query, candidate_k)
            if bm25_results:
                ranked_lists.append(bm25_results)

        # ── Semantic search ───────────────────────────────────────────────────
        if use_semantic:
            semantic_results = self._semantic_search(query, candidate_k, filters)
            if semantic_results:
                ranked_lists.append(semantic_results)

        if not ranked_lists:
            return []

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = self._rrf_fusion(*ranked_lists)

        # ── Assemble result chunks ────────────────────────────────────────────
        results = []
        for chunk_id, rrf_score in fused[:top_k]:
            # Local mode: full chunk from in-memory lookup
            chunk = self.chunk_lookup.get(chunk_id)
            if chunk:
                chunk_copy = chunk.copy()
            else:
                # Cloud mode: reconstruct from Pinecone metadata cache
                # _semantic_search populated this cache during the query
                chunk_copy = self._pinecone_meta_cache.get(chunk_id, {
                    "chunk_id":     chunk_id,
                    "text":         "",
                    "text_original": "",
                    "authors":      "",
                    "year":         "",
                    "journal":      "",
                    "section":      "",
                    "pmid":         "",
                    "citation":     "",
                })

            chunk_copy["retrieval_score"] = round(rrf_score, 6)
            results.append(chunk_copy)

        return results