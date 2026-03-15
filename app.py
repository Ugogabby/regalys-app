"""
app.py — REgalys
─────────────────────────────────────────────────────────────────────────────
Real-world Evidence Generation and Analysis Insights

RETRIEVAL ARCHITECTURE (state-of-the-art, 2025):
─────────────────────────────────────────────────
1. HyDE (Hypothetical Document Embedding — Gao et al. 2022, arXiv:2212.10496)
   Generates a hypothetical methods-paper excerpt to guide retrieval into
   document space rather than query space. Substantially improves methodology
   paper recall vs. raw query embedding. Cost: ~$0.001 (Claude Haiku).

2. Multi-query retrieval (RAG-Fusion — Raudaschl 2023)
   Decomposes query into 3 sub-queries, retrieves independently, fuses with RRF.
   Covers all facets of complex pharmacoepidemiology questions.

3. PICO extraction + study quality assessment
   Claude Haiku extracts Population/Intervention/Comparator/Outcome and
   pharmacoepidemiology quality flags per chunk. Outputs HTA evidence table
   and downloadable Word report. Cost: ~$0.001 per chunk.

4. Live Semantic Scholar augmentation (optional)
   Real-time search of Semantic Scholar for papers published after KB freeze.
   Deduplicated by PMID before entering the synthesis context.

DEDUPLICATION:
   After RRF fusion, only the highest-scoring chunk per unique PMID is kept.
   This ensures each article contributes at most one passage to the synthesis
   context, forcing retrieval diversity across papers rather than depth within
   a single paper. For books/non-PMID sources, deduplication is by title.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import streamlit as st
import anthropic
import pandas as pd

from retrieval.retriever  import HybridRetriever
from retrieval.reranker   import CohereReranker
from pico_extractor       import PICOEnricher
from live_search          import LiveSearchAugmenter
from evidence_report      import build_word_report
from config               import cfg


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
    .main-header { font-size: 2rem; font-weight: 700; color: #1f4e79; margin-bottom: 0.2rem; }
    .sub-header  { font-size: 1rem; color: #555; margin-bottom: 2rem; }
    .chunk-meta  { color: #888; font-size: 0.75rem; margin-bottom: 0.3rem; }
    .score-badge   { background: #e8f0fe; color: #1f4e79; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
    .section-badge { background: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 4px; }
    .live-badge    { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 4px; }
    .stButton > button { background-color: #1f4e79; color: white; border-radius: 6px; padding: 0.5rem 2rem; font-weight: 600; }
    .timing-bar { font-size: 0.75rem; color: #888; margin-top: 0.5rem; }
    /* ── ARIA personal knowledge badge styles (added 2026-03-15) ─────────────
       Green palette to contrast with score-badge (blue) and section-badge (yellow).
       .aria-badge      — "👤 Personal" pill label
       .aria-file-badge — source filename pill (e.g. ltcdc_sas_extraction.sas)
       .aria-score-badge — retrieval score, replaces blue .score-badge for ARIA chunks
    ── */
    .aria-badge      { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; margin-left: 4px; border: 1px solid #c3e6cb; }
    .aria-file-badge { background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 12px; font-size: 0.68rem; font-weight: 600; margin-left: 4px; }
    .aria-score-badge{ background: #c8e6c9; color: #1b5e20; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever():
    return HybridRetriever()

@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    return CohereReranker()

@st.cache_resource(show_spinner="Initializing PICO enrichment module...")
def load_enricher():
    return PICOEnricher()

@st.cache_resource(show_spinner="Connecting to Semantic Scholar...")
def load_live_augmenter():
    return LiveSearchAugmenter()


# ─────────────────────────────────────────────────────────────────────────────
# HyDE: Hypothetical Document Embedding
# ─────────────────────────────────────────────────────────────────────────────
def generate_hyde_query(user_query: str) -> tuple[str, list[str]]:
    """
    Implements HyDE (Gao et al. 2022, arXiv:2212.10496).
    Generates a hypothetical methods-paper excerpt + 3 sub-queries via Claude Haiku.
    Returns (hyde_excerpt, [sub_query_1, sub_query_2, sub_query_3]).
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
        raw  = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return data.get("hyde_excerpt", user_query), data.get("sub_queries", [user_query])
    except Exception:
        return user_query, [user_query]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query retrieval with RRF fusion + per-article deduplication
# ─────────────────────────────────────────────────────────────────────────────
def _dedup_key(chunk: dict) -> str:
    """
    Returns a deduplication key for a chunk.
    - Uses PMID for PubMed-indexed articles (most reliable unique identifier).
    - Falls back to normalized title for books and non-PMID sources.
    This ensures at most one chunk per article enters the synthesis context.
    """
    pmid = str(chunk.get("pmid", "")).strip()
    if pmid and not pmid.startswith("BOOK_") and pmid not in ("", "None", "nan"):
        return f"pmid:{pmid}"
    # Fallback: normalize title (lowercase, strip whitespace)
    title = str(chunk.get("title", chunk.get("text", "")[:60])).lower().strip()
    return f"title:{title}"


def multi_query_retrieve(
    retriever, hyde_excerpt, sub_queries, user_query,
    candidate_k, pinecone_filter, use_bm25, use_semantic,
) -> list[dict]:
    """
    Retrieves candidates using 5 parallel queries (HyDE + 3 sub-queries + original)
    and fuses results with Reciprocal Rank Fusion (k=60).

    Post-fusion deduplication: only the highest RRF-scored chunk per unique
    article (keyed by PMID, or title for books) is retained. This prevents
    a single highly-cited paper from dominating the retrieved context with
    multiple redundant passages, and forces diversity across articles.
    """
    RRF_K       = 60
    all_queries = [hyde_excerpt] + sub_queries + [user_query]
    all_ranked  = []
    chunk_pool  = {}

    for q in all_queries:
        results = retriever.retrieve(
            query=q, top_k=candidate_k, filters=pinecone_filter,
            use_bm25=use_bm25, use_semantic=use_semantic,
        )
        ranked = []
        for chunk in results:
            cid = chunk.get("chunk_id", chunk.get("pmid", "") + str(chunk.get("text", "")[:30]))
            chunk_pool[cid] = chunk
            ranked.append((cid, chunk.get("retrieval_score", 0.0)))
        if ranked:
            all_ranked.append(ranked)

    # ── RRF fusion ────────────────────────────────────────────────────────────
    rrf_scores = {}
    for ranked_list in all_ranked:
        for rank, (cid, _) in enumerate(ranked_list, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])

    # Attach RRF scores back to chunks
    scored_chunks = []
    for cid, score in fused:
        if cid in chunk_pool:
            chunk = chunk_pool[cid].copy()
            chunk["retrieval_score"] = round(score, 6)
            scored_chunks.append(chunk)

    # ── Per-article deduplication ─────────────────────────────────────────────
    # Iterate RRF-ranked chunks; keep only the first (highest-scoring) chunk
    # seen for each unique article. Later chunks from the same article are
    # discarded. This runs BEFORE reranking so Cohere sees a diverse candidate
    # set rather than multiple passages from the same paper.
    seen_articles: dict[str, bool] = {}
    deduplicated: list[dict] = []

    for chunk in scored_chunks:
        key = _dedup_key(chunk)
        if key not in seen_articles:
            seen_articles[key] = True
            deduplicated.append(chunk)

    return deduplicated


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis prompt + answer
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
        is_live  = chunk.get("is_live_result", False)
        pmid_str = f" | PMID: {pmid}" if pmid and not str(pmid).startswith("BOOK_") else ""
        live_str = " [LIVE — recent literature]" if is_live else ""
        context_blocks.append(
            f"[{i}] {authors} ({year}). {title}. {journal}{pmid_str}{live_str}\n"
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

{{query}}

You have been provided with {{len(chunks)}} highly relevant passages retrieved from a curated pharmacoepidemiology literature database of 3,894 peer-reviewed papers. Each passage is from a DISTINCT article — one chunk per paper. Chunks marked [LIVE] are from recent papers retrieved in real time. These are your ONLY permitted sources of factual claims. Every assertion must be supported by at least one citation. If passages do not contain sufficient evidence, explicitly state: "The retrieved literature does not provide direct evidence on [specific aspect]" — do NOT speculate.

RETRIEVED LITERATURE:
{{context}}

Structure your response as follows:

## Overview
High-level conceptual summary with foundational citations.

## Core Methodology / Evidence
Step-by-step guidance or evidence synthesis. Every claim immediately followed by [N].

## Comparative Approaches (if applicable)
| Approach | When to Use | Key Assumption | Limitation | Source |
|---|---|---|---|---|

## Implementation Guidance
Concrete, actionable guidance for real-world claims data (Medicaid/Medicare/T-MSIS).

## Key Assumptions and Validity Conditions
Every critical assumption, with citations.

## Caveats, Limitations, and Areas of Ongoing Debate
What remains contested; cite disagreements explicitly.

## Bottom Line for Practice
Concise actionable summary for a pharmacoepidemiology researcher.

CITATION RULES: Every factual claim must end with [N]. Never cite sources for claims they don't support. Never introduce information absent from retrieved passages.
FORMAT: markdown headers, bullets, bold key terms, tables for comparisons. PhD level. Start directly with Overview."""


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
# ARIA integration: _is_aria_chunk() detects personal knowledge chunks (2026-03-15).
# aria/embedder.py sets source_type="personal" and IDs as "aria-{hash}-{pos}".
# Published REgalys literature never carries these markers.
def _is_aria_chunk(chunk: dict) -> bool:
    """
    Returns True if this chunk came from ARIA personal knowledge ingestion.

    Three independent detection signals (any one is sufficient):
      1. source_type == "personal"        — set by aria/embedder.py on every upsert
      2. pmid starts with "aria-"         — ARIA chunk IDs: aria-{md5hash}-{position}
      3. journal contains "ARIA Personal" — set in aria/extractors/_base.py build_chunk()

    Three signals for robustness: in cloud mode (no local all_chunks.json),
    Pinecone metadata may be partially truncated — at least one signal survives.
    """
    return (
        chunk.get("source_type") == "personal"
        or str(chunk.get("pmid", "")).startswith("aria-")
        or "ARIA Personal Knowledge" in chunk.get("journal", "")
    )


def render_chunk_card(chunk: dict, index: int):
    authors  = chunk.get("authors",  "Unknown")[:50]
    year     = chunk.get("year",     "????")
    journal  = chunk.get("journal",  "Unknown")[:40]
    section  = chunk.get("section",  "")
    pmid     = chunk.get("pmid",     "")
    filename = chunk.get("filename", "")   # ARIA: source filename e.g. ltcdc_sas_extraction.sas
    text     = chunk.get("text_original", chunk.get("text", ""))
    is_live  = chunk.get("is_live_result", False)
    is_aria  = _is_aria_chunk(chunk)      # ARIA: True if personal knowledge chunk

    rerank_score    = chunk.get("rerank_score",    None)
    retrieval_score = chunk.get("retrieval_score", None)
    score_str = ""
    if rerank_score    is not None: score_str += f"rerank: {rerank_score:.3f}"
    if retrieval_score is not None: score_str += f"  rrf: {retrieval_score:.4f}"

    # ARIA: personal chunks get 👤 prefix; live chunks keep LIVE prefix; published unchanged
    if is_aria:
        label = f"[{index}] 👤 {authors} ({year})"
    else:
        label = f"[{'LIVE ' if is_live else ''}{index}] {authors} ({year})"

    with st.expander(label, expanded=False):
        if is_aria:
            # ── ARIA personal knowledge — green badge treatment ────────────
            # .aria-badge: "👤 Personal" pill | .aria-file-badge: filename pill
            # .aria-score-badge: green score (replaces blue .score-badge)
            meta_html = (
                f'<span class="chunk-meta">{journal}</span>'
                f'<span class="aria-badge">👤 Personal</span>'
            )
            if filename:
                # Show source filename so you know which file this chunk came from
                meta_html += f'<span class="aria-file-badge">📄 {filename}</span>'
            meta_html += f'<br><span class="aria-score-badge">{score_str}</span>'
            st.markdown(meta_html, unsafe_allow_html=True)
            # Show ARIA file category (sas_code, pdf_document, etc.)
            # Personal files have no PubMed ID — no link shown
            file_type = chunk.get("file_type", section)
            if file_type:
                st.caption(f"Source type: {file_type}")
        else:
            # ── Published literature — original REgalys design, unchanged ─
            live_badge = '<span class="live-badge">🌐 Live</span>' if is_live else ""
            st.markdown(
                f'<span class="chunk-meta">{journal}</span>'
                f'<span class="section-badge">{section}</span>'
                f'{live_badge}<br><span class="score-badge">{score_str}</span>',
                unsafe_allow_html=True,
            )
            # Exclude textbooks (BOOK_) and ARIA chunks (aria-) from PubMed link
            if pmid and not str(pmid).startswith("BOOK_") and not str(pmid).startswith("aria-"):
                st.markdown(f'[PubMed {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)')

        st.markdown("---")
        preview = text[:400] + "..." if len(text) > 400 else text
        st.write(preview)
        if len(text) > 400:
            with st.expander("Show full chunk", expanded=False):
                st.write(text)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence table display
# ─────────────────────────────────────────────────────────────────────────────
def render_evidence_table(evidence_table: list[dict]):
    if not evidence_table:
        st.info("No medium/high-confidence PICO extractions for this query.")
        return

    df = pd.DataFrame(evidence_table)
    display_cols = [c for c in [
        "Title", "Year", "Study Design", "Population",
        "Intervention", "Comparator", "Outcome",
        "Active Comparator", "New User Design", "Overall Quality",
    ] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    with st.expander("Show full evidence table (all columns including quality flags)", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Evidence Table (.csv)",
        data=csv,
        file_name=f"regalys_evidence_table_{int(time.time())}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(chunks=None, hyde_excerpt=None, sub_queries=None, quality_summary=None):
    with st.sidebar:
        st.markdown("### 🔬 Knowledge Base")
        st.markdown("**Papers:** 3,894 | **Chunks:** 302,377+")
        st.markdown("**👤 Personal files:** ARIA enabled")  # ARIA: personal KB active
        st.markdown("**Embeddings:** Voyage AI voyage-3")
        st.markdown("**Reranker:** Cohere Rerank 3")
        st.markdown("**Retrieval:** HyDE + Multi-query RRF")
        st.markdown("---")

        st.markdown("### ⚙️ Search Options")
        use_bm25     = st.checkbox("BM25 keyword search", value=True,  key="cb_bm25")
        use_semantic = st.checkbox("Semantic search",     value=True,  key="cb_semantic")
        use_rerank   = st.checkbox("Cohere reranking",    value=True,  key="cb_rerank")
        use_hyde     = st.checkbox(
            "HyDE retrieval ✨", value=True, key="cb_hyde",
            help="Hypothetical Document Embedding — generates an ideal methods excerpt to guide retrieval. Adds ~2s, costs ~$0.001.",
        )
        top_k = st.slider("Chunks to retrieve", 4, 16, 16, key="slider_topk")
        section_filter = st.multiselect(
            "Filter by section",
            ["methods", "results", "discussion", "abstract", "introduction", "conclusion"],
            default=[], key="ms_section",
        )

        st.markdown("---")
        st.markdown("### 🌐 Live Augmentation")
        use_live = st.checkbox(
            "Augment with live Semantic Scholar search", value=False, key="cb_live",
            help="Fetches recent papers published after the knowledge base was built.",
        )
        live_year = st.selectbox(
            "Include papers from", options=[2025, 2024, 2023], index=1,
            key="sel_live_year", disabled=not use_live,
        )
        live_n = st.slider("Max fresh papers to add", 1, 10, 5, key="slider_live_n", disabled=not use_live)

        st.markdown("---")
        st.markdown("### 🔍 PICO & Quality")
        run_pico = st.checkbox(
            "Extract PICO + quality flags", value=True, key="cb_pico",
            help="Extracts PICO elements and pharmacoepidemiology quality flags. Enables evidence table and Word report. Adds ~3-8s.",
        )

        st.markdown("---")

        # HyDE debug info
        if hyde_excerpt and hyde_excerpt != st.session_state.get("query", ""):
            with st.expander("🧠 HyDE Query", expanded=False):
                st.caption("Hypothetical document excerpt used for retrieval:")
                st.write(hyde_excerpt[:400] + "..." if len(hyde_excerpt) > 400 else hyde_excerpt)
            if sub_queries:
                with st.expander("🔀 Sub-queries", expanded=False):
                    for i, q in enumerate(sub_queries, 1):
                        st.caption(f"Q{i}: {q}")

        # Quality summary
        if quality_summary and quality_summary.get("total_assessed", 0) > 0:
            st.markdown("### 📊 Quality Assessment")
            n = quality_summary["total_assessed"]
            st.markdown(f"**{n} papers assessed**")
            for label, key in [
                ("Active comparator", "active_comparator"),
                ("New user design",   "new_user_design"),
                ("Validated outcome", "validated_outcome"),
                ("Competing events",  "competing_events_handled"),
                ("Immortal time ✓",   "immortal_time_protected"),
            ]:
                st.caption(f"{label}: **{quality_summary.get(key, '—')}**")
            dist = quality_summary.get("quality_distribution", {})
            if dist:
                st.caption(f"Strong: {dist.get('strong',0)} · Moderate: {dist.get('moderate',0)} · Weak: {dist.get('weak',0)}")
            st.markdown("---")

        if chunks:
            # ARIA: count personal vs literature chunks and show split when personal present
            personal_count   = sum(1 for c in chunks if _is_aria_chunk(c))
            literature_count = len(chunks) - personal_count
            st.markdown(f"### 📄 Retrieved Chunks ({len(chunks)})")
            if personal_count > 0:
                # Show breakdown only when ARIA personal chunks are retrieved
                st.caption(f"📚 Literature: {literature_count} · 👤 Personal: {personal_count}")
            for i, chunk in enumerate(chunks, start=1):
                render_chunk_card(chunk, i)

    return use_bm25, use_semantic, use_rerank, use_hyde, top_k, section_filter, use_live, live_year, live_n, run_pico


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for key, default in [
        ("query",           ""),
        ("chunks",          []),
        ("answer",          ""),
        ("timing",          {}),
        ("trigger_search",  False),
        ("hyde_excerpt",    None),
        ("sub_queries",     None),
        ("evidence_table",  []),
        ("quality_summary", {}),
        ("report_bytes",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(
        '<div class="main-header">🔬 REgalys</div>'
        '<div class="sub-header">Real-world Evidence Generation and Analysis Insights — '
        '3,894 papers · 302,377 chunks · TTE · Competing Events · '
        'Opioid Epidemiology · Causal Inference · PICO Extraction · Live Search</div>',
        unsafe_allow_html=True,
    )

    retriever      = load_retriever()
    reranker       = load_reranker()
    enricher       = load_enricher()
    live_augmenter = load_live_augmenter()

    (use_bm25, use_semantic, use_rerank, use_hyde,
     top_k, section_filter,
     use_live, live_year, live_n,
     run_pico) = render_sidebar(
        chunks=st.session_state.chunks or None,
        hyde_excerpt=st.session_state.hyde_excerpt,
        sub_queries=st.session_state.sub_queries,
        quality_summary=st.session_state.quality_summary or None,
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
        if st.button("TTE design",            key="btn_tte"):    set_example("TTE design")
        if st.button("Competing events",       key="btn_comp"):   set_example("Competing events")
    with col2:
        if st.button("Opioid + gabapentinoid", key="btn_opioid"): set_example("Opioid + gaba")
        if st.button("CCW method",             key="btn_ccw"):    set_example("CCW method")
    with col3:
        if st.button("Claims data coding",     key="btn_claims"): set_example("Claims data")
        if st.button("Active comparator",      key="btn_ac"):     set_example("Active comparator")

    st.markdown("#### Your query")
    user_query = st.text_area(
        label="Query", value=st.session_state.query,
        placeholder="e.g. How do I implement CCW with competing events and time-varying confounding in Medicaid claims?",
        height=80, label_visibility="collapsed", key="ta_query",
    )

    search_clicked = st.button("🔍 Search & Synthesize", type="primary", key="btn_search")
    should_search  = search_clicked or st.session_state.trigger_search
    active_query   = st.session_state.query if st.session_state.trigger_search else user_query

    if should_search and active_query.strip():
        st.session_state.trigger_search  = False
        st.session_state.query           = active_query
        st.session_state.evidence_table  = []
        st.session_state.quality_summary = {}
        st.session_state.report_bytes    = None

        pinecone_filter = {"section": {"$in": section_filter}} if section_filter else None
        # Increase candidate_k to compensate for deduplication reducing pool size.
        # With deduplication, many candidates will be dropped, so we retrieve more
        # upfront to ensure top_k diverse articles reach the reranker.
        candidate_k = top_k * 10

        # Stage 1: HyDE + sub-query generation
        hyde_excerpt, sub_queries, hyde_time = active_query, [active_query], 0.0
        if use_hyde:
            with st.spinner("🧠 Generating HyDE retrieval query (Claude Haiku)..."):
                t_hyde = time.time()
                hyde_excerpt, sub_queries = generate_hyde_query(active_query)
                hyde_time = time.time() - t_hyde
        st.session_state.hyde_excerpt = hyde_excerpt
        st.session_state.sub_queries  = sub_queries

        # Stage 2: Multi-query retrieval with RRF fusion + per-article deduplication
        with st.spinner("Searching 302,377 chunks across multiple query facets..."):
            t0     = time.time()
            chunks = multi_query_retrieve(
                retriever=retriever, hyde_excerpt=hyde_excerpt,
                sub_queries=sub_queries, user_query=active_query,
                candidate_k=candidate_k, pinecone_filter=pinecone_filter,
                use_bm25=use_bm25, use_semantic=use_semantic,
            )
            retrieval_time = time.time() - t0

        # Stage 3: Live Semantic Scholar augmentation
        if use_live and live_augmenter.is_available:
            with st.spinner(f"Fetching fresh literature from Semantic Scholar (>={live_year})..."):
                chunks = live_augmenter.augment(
                    query=active_query, existing_chunks=chunks,
                    top_n_fresh=live_n, year_from=live_year,
                )

        # Stage 4: Cohere reranking against HyDE excerpt
        # Reranking against the HyDE excerpt (not the user query) so Cohere
        # scores methodology papers correctly — core to the HyDE architecture.
        # Deduplication has already run, so Cohere sees one chunk per article.
        if use_rerank and chunks:
            with st.spinner("Reranking with Cohere..."):
                t1          = time.time()
                chunks      = reranker.rerank(hyde_excerpt, chunks, top_n=top_k)
                rerank_time = time.time() - t1
        else:
            chunks, rerank_time = chunks[:top_k], 0.0

        st.session_state.chunks = chunks

        # Stage 5: PICO extraction + quality assessment
        evidence_table, quality_summary, pico_time = [], {}, 0.0
        if run_pico and chunks:
            with st.spinner(f"Extracting PICO + quality flags from {len(chunks)} chunks (Claude Haiku)..."):
                t_pico          = time.time()
                enriched        = enricher.enrich(chunks)
                evidence_table  = enricher.to_evidence_table(enriched, min_confidence="medium")
                quality_summary = enricher.quality_summary(enriched)
                pico_time       = time.time() - t_pico
            st.session_state.evidence_table  = evidence_table
            st.session_state.quality_summary = quality_summary

        # Stage 6: Claude Sonnet synthesis — always uses original user query
        with st.spinner("Synthesizing comprehensive answer with Claude Sonnet..."):
            t2       = time.time()
            answer   = synthesize_answer(active_query, chunks)
            llm_time = time.time() - t2

        st.session_state.answer = answer
        st.session_state.timing = {
            "hyde": hyde_time, "retrieval": retrieval_time,
            "rerank": rerank_time, "pico": pico_time,
            "llm": llm_time, "chunks": len(chunks),
        }

        # Stage 7: Pre-build Word report
        if run_pico:
            try:
                st.session_state.report_bytes = build_word_report(
                    query=active_query, answer=answer,
                    evidence_table=evidence_table, quality_summary=quality_summary,
                    chunks=chunks, timing=st.session_state.timing,
                )
            except Exception as e:
                print(f"  [Report] {e}")

        st.rerun()

    elif should_search:
        st.warning("Please enter a query.")

    # ── Output: three tabs ────────────────────────────────────────────────────
    if st.session_state.answer:
        tab1, tab2, tab3 = st.tabs(["📝 Synthesis", "📋 Evidence Table", "🏆 Quality Summary"])

        with tab1:
            st.markdown("### Answer")
            st.divider()
            st.markdown(st.session_state.answer)
            st.divider()

            t     = st.session_state.timing
            total = sum(t.get(k, 0) for k in ["hyde", "retrieval", "rerank", "pico", "llm"])
            hyde_str = f"HyDE: {t.get('hyde',0):.1f}s · " if use_hyde else ""
            pico_str = f"PICO: {t.get('pico',0):.1f}s · " if t.get("pico",0) > 0 else ""
            st.caption(
                f"⏱ {hyde_str}Retrieval: {t.get('retrieval',0):.1f}s · "
                f"Rerank: {t.get('rerank',0):.1f}s · {pico_str}"
                f"LLM: {t.get('llm',0):.1f}s · Total: {total:.1f}s · "
                f"{t.get('chunks',0)} chunks used · Max response: 4,096 tokens"
            )

            if st.session_state.report_bytes:
                st.download_button(
                    label="📥 Download Evidence Report (.docx)",
                    data=st.session_state.report_bytes,
                    file_name=f"regalys_evidence_report_{int(time.time())}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    help="Full synthesis + PICO evidence table + quality summary + references as Word document",
                )

        with tab2:
            st.markdown("#### PICO Evidence Table")
            st.caption(
                "Structured PICO elements and pharmacoepidemiology quality flags extracted "
                "from retrieved papers. Filtered to medium and high confidence extractions. "
                "Compatible with HTA dossier evidence table format (NICE, ICER, G-BA)."
            )
            if st.session_state.evidence_table:
                render_evidence_table(st.session_state.evidence_table)
            else:
                st.info(
                    "Enable 'Extract PICO + quality flags' in the sidebar to populate this table."
                    if not run_pico else
                    "No medium/high-confidence PICO extractions found for this query."
                )

        with tab3:
            st.markdown("#### Study Quality Summary")
            st.caption(
                "Aggregate quality statistics across retrieved papers. "
                "Flags follow pharmacoepidemiology standards: active comparator new-user design, "
                "validated outcome definitions, adequate confounding control."
            )
            qs = st.session_state.quality_summary
            if qs and qs.get("total_assessed", 0) > 0:
                n    = qs["total_assessed"]
                dist = qs.get("quality_distribution", {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Papers assessed",  n)
                    st.metric("Strong quality",   dist.get("strong",   0))
                with c2:
                    st.metric("Active comparator", qs.get("active_comparator", "—"))
                    st.metric("Moderate quality",  dist.get("moderate", 0))
                with c3:
                    st.metric("New user design",  qs.get("new_user_design", "—"))
                    st.metric("Weak quality",     dist.get("weak", 0))
                st.markdown("---")
                st.markdown("**Confounding adjustment methods in retrieved papers:**")
                conf = qs.get("confounding_methods", {})
                if conf:
                    st.dataframe(
                        pd.DataFrame([{"Method": k, "Count": v}
                                      for k, v in sorted(conf.items(), key=lambda x: -x[1])]),
                        use_container_width=False, hide_index=True,
                    )
                st.markdown("---")
                for label, key in [
                    ("Validated outcome definition",  "validated_outcome"),
                    ("Competing events addressed",    "competing_events_handled"),
                    ("Immortal time bias protected",  "immortal_time_protected"),
                ]:
                    st.caption(f"**{label}:** {qs.get(key, '—')}")
            else:
                st.info(
                    "Enable 'Extract PICO + quality flags' in the sidebar."
                    if not run_pico else "No quality data available for this query."
                )

    st.markdown("---")
    st.caption(
        "REgalys · Real-world Evidence Generation and Analysis Insights · "
        "Built by Ugochukwu Ezigbo BPharm, MHA · "
        "PhD Candidate, University of Pittsburgh · Pharmaceutical Outcomes & Policy · "
        "👤 ARIA personal knowledge enabled · "   # ARIA: footer indicator
        "[GitHub](https://github.com/Ugogabby/regalys)"
    )


if __name__ == "__main__":
    main()