"""
retrieval/reranker.py
──────────────────────
Cohere Rerank 3 cross-encoder reranker.

Why reranking?
  BM25 and semantic search are both "bi-encoders" — they encode the query
  and each document independently, then compare vectors. This is fast but
  loses nuance: the model never sees query and document together.

  A cross-encoder (reranker) sees the query AND document simultaneously,
  scoring their relevance jointly. This is much more accurate but too slow
  to run over 297k chunks. So we:
    1. Use fast hybrid retrieval to get top 24 candidates
    2. Use the slow-but-accurate reranker to re-score just those 24
    3. Return the top 8 after reranking

  In pharmacoepidemiology this matters enormously. A query like:
    "competing events in opioid overdose studies"
  might retrieve a chunk about "competing risks in cancer trials" high up
  via semantic search. The reranker demotes it because it sees the full
  query + chunk together and scores the mismatch.

Cohere Rerank 3:
  State-of-the-art cross-encoder. Scores each (query, document) pair
  on a relevance scale. Returns scores between 0 and 1.
  Much more accurate than cosine similarity for final ranking.
"""

import cohere
from config import cfg


class CohereReranker:
    """
    Wraps Cohere Rerank 3 for reranking retrieved chunks.

    Usage:
        reranker = CohereReranker()
        reranked = reranker.rerank(query, chunks, top_n=8)
    """

    def __init__(self):
        """Initializes the Cohere client."""
        if not cfg.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not set in .env")
        self.client = cohere.Client(api_key=cfg.COHERE_API_KEY)
        self.model  = "rerank-english-v3.0"

    def rerank(
        self,
        query:  str,
        chunks: list[dict],
        top_n:  int = None,
    ) -> list[dict]:
        """
        Reranks a list of retrieved chunks using Cohere cross-encoder.

        Args:
            query:  the search query
            chunks: list of chunk dicts from HybridRetriever.retrieve()
            top_n:  number of chunks to return after reranking
                    defaults to cfg.TOP_K

        Returns:
            reranked list of chunk dicts with added 'rerank_score' field,
            sorted by rerank_score descending

        Notes:
            - Cohere Rerank accepts up to 1000 documents per request
            - We typically pass 24 candidates and return top 8
            - Each chunk is represented to Cohere by its text_original field
              (the raw chunk without the context prefix) for cleaner scoring
        """
        if top_n is None:
            top_n = cfg.TOP_K

        if not chunks:
            return []

        # Extract the text to send to Cohere
        # Use text_original (without context prefix) for accurate scoring
        documents = [
            c.get("text_original", c.get("text", ""))
            for c in chunks
        ]

        try:
            response = self.client.rerank(
                model     = self.model,
                query     = query,
                documents = documents,
                top_n     = top_n,
            )

            # Re-order chunks according to Cohere's ranking
            reranked = []
            for result in response.results:
                chunk = chunks[result.index].copy()
                chunk["rerank_score"]    = round(result.relevance_score, 4)
                chunk["rerank_position"] = result.index  # original position
                reranked.append(chunk)

            return reranked

        except Exception as e:
            # If reranking fails, return original order
            print(f"⚠ Cohere reranking failed: {e} — returning original order")
            return chunks[:top_n]


def rerank_results(
    query:  str,
    chunks: list[dict],
    top_n:  int = None,
) -> list[dict]:
    """
    Module-level convenience function for reranking.

    Creates a fresh CohereReranker instance and reranks in one call.
    Use this for one-off reranking. For repeated queries (e.g. in the app),
    instantiate CohereReranker once and reuse it.

    Args:
        query:  search query
        chunks: chunks from HybridRetriever
        top_n:  number to return

    Returns:
        reranked chunks with rerank_score field
    """
    reranker = CohereReranker()
    return reranker.rerank(query, chunks, top_n)