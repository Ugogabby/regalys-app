"""
app.py — REgalys
─────────────────────────────────────────────────────────────────────────────
Real-world Evidence Generation and Analysis Insights

RETRIEVAL ARCHITECTURE (state-of-the-art, 2025):
─────────────────────────────────────────────────
Standard RAG fails on domain-specific queries because:
  1. A clinical query ("gabapentinoid overdose TTE") retrieves clinical papers
     (Gomes, Evoy) instead of methodology papers (Hernán, Yoshida CCW) because
     the embedding space scores drug/outcome terms higher than method terms.
  2. A single query vector misses concepts described with different vocabulary.
  3. Cohere reranker amplifies the problem by rescoring against the same
     clinical query.

This version implements three state-of-the-art solutions:

1. HyDE (Hypothetical Document Embedding — Gao et al. 2022, arXiv:2212.10496)
   ─────────────────────────────────────────────────────────────────────────
   Instead of embedding the user query directly, Claude generates a short
   hypothetical "ideal methodology paper excerpt" that would perfectly answer
   the question. That excerpt is embedded and used for retrieval.
   
   Why it works: The hypothetical excerpt lives in document space (same
   distribution as the indexed chunks), not query space. A hypothetical
   methods paper about CCW with competing events will retrieve CCW methods
   papers, regardless of how the user phrased their question.
   
   Cost: ~$0.001 per query (Claude Haiku, ~200 tokens).

2. Multi-query retrieval (RAG-Fusion — Raudaschl 2023)
   ─────────────────────────────────────────────────────────────────────────
   Claude decomposes the user query into 3 sub-queries targeting different
   facets (methods, clinical context, implementation). Each sub-query is
   embedded and retrieved independently. Results are fused with RRF.
   
   Why it works: Different aspects of a complex question live in different
   parts of the vector space. Multi-query retrieval ensures all facets are
   covered, not just the dominant semantic cluster.

3. Larger + diverse candidate pool
   ─────────────────────────────────────────────────────────────────────────
   Retrieve top_k * 6 candidates (vs * 3 before) before reranking, ensuring
   methodology papers are in the pool even if clinical papers rank higher
   in raw semantic search. Cohere then reranks the full pool against the
   HyDE query — which is already methodology-framed — giving methods papers
   their correct relevance signal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import streamlit as st
import anthropic

from retrieval.retriever import HybridRetriever
from retrieval.reranker  import CohereReranker
from config import cfg


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "REgalys — Real-world Evidence Generation and Analysis Insights",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .chunk-meta {
        color: #888;
        font-size: 0.75rem;
        margin-bottom: 0.3rem;
    }
    .score-badge {
        background: #e8f0fe;
        color: #1f4e79;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .section-badge {
        background: #fff3cd;
        color: #856404;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 4px;
    }
    .hyde-badge {
        background: #e6f4ea;
        color: #1e7e34;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 4px;
    }
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .timing-bar {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever():
    return HybridRetriever()

@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    return CohereReranker()


# ─────────────────────────────────────────────────────────────────────────────
# HyDE: Hypothetical Document Embedding
# ─────────────────────────────────────────────────────────────────────────────
def generate_hyde_query(user_query: str) -> tuple[str, list[str]]:
    """
    Implements HyDE (Hypothetical Document Embedding, Gao et al. 2022).

    Generates:
      1. A hypothetical pharmacoepidemiology paper excerpt that would perfectly
         answer the user's question — used as the primary retrieval vector.
         This excerpt lives in document space, so it retrieves documents more
         accurately than a raw query vector.

      2. Three sub-queries targeting different facets of the question —
         used for multi-query retrieval (RAG-Fusion).

    Uses Claude Haiku for speed and cost (~$0.001 per call).

    Returns:
        (hyde_excerpt, [sub_query_1, sub_query_2, sub_query_3])
    """
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    prompt = f"""You are an expert pharmacoepidemiology methodologist specializing in:
- Target trial emulation (TTE) and sequential trial emulation
- Clone-censor-weight (CCW) methodology
- Competing events (cause-specific hazards, Fine-Gray, estimands)
- Inverse probability of censoring weighting (IPCW)
- Time-varying confounding in administrative claims data

A researcher asked: "{user_query}"

Complete TWO tasks. Respond ONLY with valid JSON, no other text.

TASK 1 — HyDE excerpt:
Write a 200-word excerpt from the METHODS section of a pharmacoepidemiology
paper that directly addresses the METHODOLOGICAL approach in the question.

CRITICAL RULES for the excerpt:
- Focus on the CAUSAL INFERENCE METHOD (CCW, TTE, IPCW, competing events
  estimands, sequential trial emulation) — not the clinical findings
- Use precise technical vocabulary: "clone-censor-weight", "artificial censoring",
  "grace period", "IPCW", "cause-specific hazard", "subdistribution hazard",
  "time-varying confounding", "per-protocol estimand", "sequential trials"
- Describe the PROCEDURE step by step as a methods section would
- Do NOT write about drug effects, clinical outcomes, or epidemiologic findings
- This excerpt will search a database — it must match methods paper vocabulary

TASK 2 — Sub-queries (3 queries targeting different retrieval facets):
  - Sub-query 1: the core causal inference method (CCW, IPCW, TTE steps)
  - Sub-query 2: the competing events methodology (estimand choice, weight construction)
  - Sub-query 3: implementation in administrative claims data (Medicaid, T-MSIS)

Return ONLY this JSON:
{{
  "hyde_excerpt": "...",
  "sub_queries": ["...", "...", "..."]
}}"""

    try:
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 600,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw  = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        hyde_excerpt = data.get("hyde_excerpt", user_query)
        sub_queries  = data.get("sub_queries",  [user_query])
        return hyde_excerpt, sub_queries
    except Exception:
        # Graceful fallback: use original query if HyDE fails
        return user_query, [user_query]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query retrieval with RRF fusion
# ─────────────────────────────────────────────────────────────────────────────
def multi_query_retrieve(
    retriever,
    hyde_excerpt:  str,
    sub_queries:   list[str],
    user_query:    str,
    candidate_k:   int,
    pinecone_filter,
    use_bm25:      bool,
    use_semantic:  bool,
) -> list[dict]:
    """
    Retrieves candidates using four parallel queries and fuses with RRF:
      Q1: HyDE excerpt (primary — document-space retrieval)
      Q2: Sub-query 1 (core method)
      Q3: Sub-query 2 (clinical context)
      Q4: Sub-query 3 (implementation)

    All four ranked lists are fused with Reciprocal Rank Fusion.
    The user_query is also included as Q5 for recall safety.

    Returns deduplicated, RRF-fused list of chunk dicts.
    """
    RRF_K = 60

    # Run retrieval for each query
    all_queries = [hyde_excerpt] + sub_queries + [user_query]
    all_ranked  = []   # list of lists of (chunk_id, score)
    chunk_pool  = {}   # chunk_id → chunk dict (deduplicated)

    for q in all_queries:
        results = retriever.retrieve(
            query        = q,
            top_k        = candidate_k,
            filters      = pinecone_filter,
            use_bm25     = use_bm25,
            use_semantic = use_semantic,
        )
        ranked = []
        for chunk in results:
            cid = chunk.get("chunk_id", chunk.get("pmid", "") + str(chunk.get("text", "")[:30]))
            chunk_pool[cid] = chunk
            ranked.append((cid, chunk.get("retrieval_score", 0.0)))
        if ranked:
            all_ranked.append(ranked)

    # RRF fusion across all query results
    rrf_scores = {}
    for ranked_list in all_ranked:
        for rank, (cid, _) in enumerate(ranked_list, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])

    # Assemble result chunks
    results = []
    for cid, score in fused:
        if cid in chunk_pool:
            chunk = chunk_pool[cid].copy()
            chunk["retrieval_score"] = round(score, 6)
            results.append(chunk)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis prompt
# ─────────────────────────────────────────────────────────────────────────────
def build_synthesis_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        authors  = chunk.get("authors",  "Unknown author")
        year     = chunk.get("year",     "????")
        journal  = chunk.get("journal",  "Unknown journal")
        title    = chunk.get("title",    "")[:150]
        section  = chunk.get("section",  "unknown section")
        pmid     = chunk.get("pmid",     "")
        text     = chunk.get("text_original", chunk.get("text", ""))
        pmid_str = f" | PMID: {pmid}" if pmid and not str(pmid).startswith("BOOK_") else ""
        context_blocks.append(
            f"[{i}] {authors} ({year}). {title}. {journal}{pmid_str}\n"
            f"    Section: {section}\n"
            f"    Text: {text}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are a world-class expert in pharmacoepidemiology, causal inference, and health outcomes research. Your expertise spans:
- Target trial emulation (TTE) and sequential trial emulation (STE)
- Competing events methodology (cause-specific hazards, Fine-Gray subdistribution hazards, estimand frameworks)
- Opioid epidemiology and opioid-sedative overdose risk (OSORD)
- Gabapentinoid-opioid concurrent use and drug interaction epidemiology
- Real-world evidence generation from administrative claims data (Medicaid, Medicare, T-MSIS)
- Causal inference methods: IPCW, clone-censor-weight (CCW), g-formula, marginal structural models
- Active comparator new user designs, immortal time bias, channeling bias, hdPS
- ICD-10 coding (T40.XX series), NDC-based drug exposure definitions, MME calculations

A researcher has asked the following question:

{query}

You have been provided with {len(chunks)} highly relevant passages retrieved from a curated pharmacoepidemiology literature database of 3,894 peer-reviewed papers. These are your ONLY permitted sources of factual claims. Every assertion you make must be supported by at least one citation from these passages. If the passages do not contain sufficient evidence to answer part of the question, explicitly state: "The retrieved literature does not provide direct evidence on [specific aspect]" — do NOT speculate or fill gaps with uncited claims.

RETRIEVED LITERATURE:
{context}

INSTRUCTIONS FOR YOUR RESPONSE:

Your response must be COMPREHENSIVE and EXHAUSTIVE — aim for the most complete answer the evidence allows. Structure your response as follows:

## Overview
Provide a high-level conceptual summary of the topic and why it matters in pharmacoepidemiology practice. Cite foundational sources.

## Core Methodology / Evidence
Provide detailed, step-by-step methodological guidance or evidence synthesis. For each major point:
- State the claim or recommendation
- Immediately cite the supporting source(s) [N]
- If multiple sources agree or disagree, synthesize them explicitly

## Comparative Approaches (if applicable)
When multiple methodological approaches exist, provide a structured comparison:
| Approach | When to Use | Key Assumption | Limitation | Source |
|---|---|---|---|---|

## Implementation Guidance
Concrete, actionable guidance for applying this in real-world claims data (Medicaid/Medicare/T-MSIS). Cite specific methodological choices from the literature.

## Key Assumptions and Validity Conditions
List every critical assumption required for valid inference, with citations for each.

## Caveats, Limitations, and Areas of Ongoing Debate
Be explicit about what remains contested, unsettled, or context-dependent. Cite disagreements between sources where they exist.

## Bottom Line for Practice
A concise, actionable summary for a pharmacoepidemiology researcher designing a study.

CITATION RULES (strictly enforced):
1. Every factual claim must end with at least one citation: [1], [2], [1,3], etc.
2. Do NOT cite sources for claims they do not support
3. Never introduce information not present in the retrieved passages
4. If evidence is thin or absent for part of the question, say so explicitly

FORMAT RULES:
- Use markdown headers, bullet points, bold for key terms, tables for comparisons
- Write at PhD level — do not oversimplify
- Start directly with the Overview section — no preamble"""


def synthesize_answer(query: str, chunks: list[dict]) -> str:
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model      = cfg.LLM_MODEL,
        max_tokens = 4096,
        messages   = [{"role": "user", "content": build_synthesis_prompt(query, chunks)}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# Chunk card
# ─────────────────────────────────────────────────────────────────────────────
def render_chunk_card(chunk: dict, index: int):
    authors  = chunk.get("authors",  "Unknown")[:50]
    year     = chunk.get("year",     "????")
    journal  = chunk.get("journal",  "Unknown")[:40]
    section  = chunk.get("section",  "")
    pmid     = chunk.get("pmid",     "")
    text     = chunk.get("text_original", chunk.get("text", ""))

    rerank_score    = chunk.get("rerank_score",    None)
    retrieval_score = chunk.get("retrieval_score", None)

    score_str = ""
    if rerank_score is not None:
        score_str += f"rerank: {rerank_score:.3f}"
    if retrieval_score is not None:
        score_str += f"  rrf: {retrieval_score:.4f}"

    with st.expander(f"[{index}] {authors} ({year})", expanded=False):
        st.markdown(
            f'<span class="chunk-meta">{journal}</span>'
            f'<span class="section-badge">{section}</span>'
            f'<br><span class="score-badge">{score_str}</span>',
            unsafe_allow_html=True,
        )
        if pmid and not str(pmid).startswith("BOOK_"):
            st.markdown(f'[PubMed {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)')
        st.markdown("---")
        preview = text[:400] + "..." if len(text) > 400 else text
        st.write(preview)
        if len(text) > 400:
            with st.expander("Show full chunk", expanded=False):
                st.write(text)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(chunks=None, hyde_excerpt=None, sub_queries=None):
    with st.sidebar:
        st.markdown("### 🔬 Knowledge Base")
        st.markdown("**Papers:** 3,894")
        st.markdown("**Chunks:** 302,377")
        st.markdown("**Vectors:** Pinecone serverless")
        st.markdown("**Embeddings:** Voyage AI voyage-3")
        st.markdown("**Reranker:** Cohere Rerank 3")
        st.markdown("**Retrieval:** HyDE + Multi-query RRF")
        st.markdown("---")

        st.markdown("### ⚙️ Search Options")
        use_bm25     = st.checkbox("BM25 keyword search", value=True,  key="cb_bm25")
        use_semantic = st.checkbox("Semantic search",     value=True,  key="cb_semantic")
        use_rerank   = st.checkbox("Cohere reranking",    value=True,  key="cb_rerank")
        use_hyde     = st.checkbox("HyDE retrieval ✨",   value=True,  key="cb_hyde",
                                   help="Hypothetical Document Embedding — generates an ideal answer excerpt to guide retrieval. Adds ~2s but substantially improves methodology paper retrieval.")
        top_k        = st.slider("Chunks to retrieve", 4, 16, 16, key="slider_topk")
        section_filter = st.multiselect(
            "Filter by section",
            ["methods", "results", "discussion", "abstract", "introduction", "conclusion"],
            default=[],
            key="ms_section",
        )
        st.markdown("---")

        # Show HyDE excerpt if available
        if hyde_excerpt and hyde_excerpt != st.session_state.get("query", ""):
            with st.expander("🧠 HyDE Query (hover to see)", expanded=False):
                st.caption("Hypothetical document excerpt used for retrieval:")
                st.write(hyde_excerpt[:400] + "..." if len(hyde_excerpt) > 400 else hyde_excerpt)
            if sub_queries:
                with st.expander("🔀 Sub-queries", expanded=False):
                    for i, q in enumerate(sub_queries, 1):
                        st.caption(f"Q{i}: {q}")

        if chunks:
            st.markdown(f"### 📄 Retrieved Chunks ({len(chunks)})")
            for i, chunk in enumerate(chunks, start=1):
                render_chunk_card(chunk, i)

    return use_bm25, use_semantic, use_rerank, use_hyde, top_k, section_filter


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for key, default in [
        ("query",          ""),
        ("chunks",         []),
        ("answer",         ""),
        ("timing",         {}),
        ("trigger_search", False),
        ("hyde_excerpt",   None),
        ("sub_queries",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(
        '<div class="main-header">🔬 REgalys</div>'
        '<div class="sub-header">'
        'Real-world Evidence Generation and Analysis Insights — '
        '3,894 papers · 302,377 chunks · TTE · Competing Events · '
        'Opioid Epidemiology · Causal Inference</div>',
        unsafe_allow_html=True,
    )

    retriever = load_retriever()
    reranker  = load_reranker()

    use_bm25, use_semantic, use_rerank, use_hyde, top_k, section_filter = render_sidebar(
        chunks       = st.session_state.chunks   or None,
        hyde_excerpt = st.session_state.hyde_excerpt,
        sub_queries  = st.session_state.sub_queries,
    )

    # Example queries
    st.markdown("#### Example queries")
    col1, col2, col3 = st.columns(3)
    examples = {
        "TTE design":        "How do you handle immortal time bias in target trial emulation using observational claims data?",
        "Competing events":  "When should I use Fine-Gray subdistribution hazard vs cause-specific hazard in pharmacoepidemiology studies with competing events?",
        "Opioid + gaba":     "What is the evidence for gabapentinoid-opioid concurrent use and overdose risk, and what study designs have been used?",
        "CCW method":        "How does the clone-censor-weight method work for sustained treatment strategies and how does it compare to sequential trial emulation?",
        "Claims data":       "How do you define opioid exposure using NDC codes and calculate morphine milligram equivalents (MME) in Medicaid claims data?",
        "Active comparator": "Why use active comparator new user design in pharmacoepidemiology and how do you implement it to reduce channeling bias?",
    }

    def set_example(key):
        st.session_state.query          = examples[key]
        st.session_state.trigger_search = True

    with col1:
        if st.button("TTE design",        key="btn_tte"):    set_example("TTE design")
        if st.button("Competing events",  key="btn_comp"):   set_example("Competing events")
    with col2:
        if st.button("Opioid + gabapentinoid", key="btn_opioid"): set_example("Opioid + gaba")
        if st.button("CCW method",             key="btn_ccw"):    set_example("CCW method")
    with col3:
        if st.button("Claims data coding", key="btn_claims"): set_example("Claims data")
        if st.button("Active comparator",  key="btn_ac"):     set_example("Active comparator")

    # Query input
    st.markdown("#### Your query")
    user_query = st.text_area(
        label            = "Query",
        value            = st.session_state.query,
        placeholder      = "e.g. How do I implement CCW with competing events and time-varying confounding in Medicaid claims?",
        height           = 80,
        label_visibility = "collapsed",
        key              = "ta_query",
    )

    search_clicked = st.button("🔍 Search & Synthesize", type="primary", key="btn_search")
    should_search  = search_clicked or st.session_state.trigger_search
    active_query   = st.session_state.query if st.session_state.trigger_search else user_query

    if should_search and active_query.strip():
        st.session_state.trigger_search = False
        st.session_state.query          = active_query

        pinecone_filter = {"section": {"$in": section_filter}} if section_filter else None
        candidate_k     = top_k * 6   # Large pool — ensures methods papers are in range

        # ── Stage 1: HyDE + sub-query generation ─────────────────────────────
        hyde_excerpt = active_query
        sub_queries  = [active_query]

        if use_hyde:
            with st.spinner("🧠 Generating HyDE retrieval query..."):
                t_hyde       = time.time()
                hyde_excerpt, sub_queries = generate_hyde_query(active_query)
                hyde_time    = time.time() - t_hyde
        else:
            hyde_time = 0

        st.session_state.hyde_excerpt = hyde_excerpt
        st.session_state.sub_queries  = sub_queries

        # ── Stage 2: Multi-query retrieval with RRF fusion ────────────────────
        with st.spinner("Searching 302,377 chunks across multiple query facets..."):
            t0    = time.time()
            chunks = multi_query_retrieve(
                retriever       = retriever,
                hyde_excerpt    = hyde_excerpt,
                sub_queries     = sub_queries,
                user_query      = active_query,
                candidate_k     = candidate_k,
                pinecone_filter = pinecone_filter,
                use_bm25        = use_bm25,
                use_semantic    = use_semantic,
            )
            retrieval_time = time.time() - t0

        # ── Stage 3: Cohere reranking against HyDE excerpt ───────────────────
        # Rerank against the HyDE excerpt (document-space query) so Cohere
        # scores methodology papers correctly, not clinical papers.
        if use_rerank and chunks:
            with st.spinner("Reranking with Cohere..."):
                t1          = time.time()
                chunks      = reranker.rerank(hyde_excerpt, chunks, top_n=top_k)
                rerank_time = time.time() - t1
        else:
            chunks, rerank_time = chunks[:top_k], 0

        st.session_state.chunks = chunks

        # ── Stage 4: Claude Sonnet synthesis ─────────────────────────────────
        # Synthesis always uses the original user query — not HyDE
        with st.spinner("Synthesizing comprehensive answer with Claude Sonnet..."):
            t2       = time.time()
            answer   = synthesize_answer(active_query, chunks)
            llm_time = time.time() - t2

        st.session_state.answer = answer
        st.session_state.timing = {
            "hyde":      hyde_time,
            "retrieval": retrieval_time,
            "rerank":    rerank_time,
            "llm":       llm_time,
            "chunks":    len(chunks),
        }
        st.rerun()

    elif should_search:
        st.warning("Please enter a query.")

    # ── Display answer ────────────────────────────────────────────────────────
    if st.session_state.answer:
        st.markdown("### Answer")
        st.divider()
        st.markdown(st.session_state.answer)
        st.divider()

        t     = st.session_state.timing
        total = sum([t.get(k, 0) for k in ["hyde", "retrieval", "rerank", "llm"]])
        hyde_str = f"HyDE: {t.get('hyde', 0):.1f}s · " if use_hyde else ""
        st.caption(
            f"⏱ {hyde_str}"
            f"Retrieval: {t.get('retrieval', 0):.1f}s · "
            f"Rerank: {t.get('rerank', 0):.1f}s · "
            f"LLM: {t.get('llm', 0):.1f}s · "
            f"Total: {total:.1f}s · "
            f"{t.get('chunks', 0)} chunks used · "
            f"Max response: 4,096 tokens"
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "REgalys · Real-world Evidence Generation and Analysis Insights · "
        "Built by Ugochukwu Ezigbo BPharm, MHA · "
        "PhD Student, University of Pittsburgh · Pharmaceutical Outcomes & Policy · "
        "[GitHub](https://github.com/Ugogabby/regalys-app)"
    )


if __name__ == "__main__":
    main()