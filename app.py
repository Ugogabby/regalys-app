"""
app/app.py
───────────
REgalys — Real-world Evidence Generation and Analysis Insights
Streamlit UI for the pharmacoepidemiology literature knowledge base.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
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


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever():
    return HybridRetriever()

@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    return CohereReranker()


# ── Synthesis prompt ──────────────────────────────────────────────────────────
def build_synthesis_prompt(query: str, chunks: list[dict]) -> str:
    """
    Builds a maximally comprehensive synthesis prompt.

    Design principles:
    - Instructs Claude to use ALL retrieved evidence, not just easy chunks
    - Requires citation of every factual claim — no uncited assertions
    - Requests high-level conceptual illustration (tables, structured comparisons)
    - Explicitly forbids hallucination — if evidence is absent, say so
    - Targets PhD-level depth: methodology, assumptions, caveats, debates
    - Asks for synthesis across sources, not just summarization of individual chunks
    - Max tokens set high (4096) to allow full comprehensive responses
    """
    # Format each chunk with full bibliographic context
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        authors  = chunk.get("authors",  "Unknown author")
        year     = chunk.get("year",     "????")
        journal  = chunk.get("journal",  "Unknown journal")
        title    = chunk.get("title",    "")[:150]
        section  = chunk.get("section",  "unknown section")
        pmid     = chunk.get("pmid",     "")
        citation = chunk.get("citation", "")
        text     = chunk.get("text_original", chunk.get("text", ""))

        # Include full text — no truncation in prompt for maximum evidence
        pmid_str = f" | PMID: {pmid}" if pmid and not str(pmid).startswith("BOOK_") else ""
        context_blocks.append(
            f"[{i}] {authors} ({year}). {title}. {journal}{pmid_str}\n"
            f"    Section: {section}\n"
            f"    Text: {text}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a world-class expert in pharmacoepidemiology, causal inference, and health outcomes research. Your expertise spans:
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
When multiple methodological approaches exist, provide a structured comparison. Use a markdown table if it aids clarity:
| Approach | When to Use | Key Assumption | Limitation | Source |
|---|---|---|---|---|

## Implementation Guidance
Concrete, actionable guidance for applying this in real-world claims data (Medicaid/Medicare/T-MSIS) or pharmacoepidemiology studies. Cite specific methodological choices from the literature.

## Key Assumptions and Validity Conditions
List every critical assumption required for valid inference, with citations for each.

## Caveats, Limitations, and Areas of Ongoing Debate
Be explicit about what remains contested, unsettled, or context-dependent in the literature. Cite disagreements between sources where they exist.

## Bottom Line for Practice
A concise, actionable summary of the key takeaways for a pharmacoepidemiology researcher designing a study.

CITATION RULES (strictly enforced):
1. Every factual claim must end with at least one citation: [1], [2], [1,3], etc.
2. Do NOT cite sources for claims they do not support — only cite what the text actually says
3. If you are uncertain whether a source supports a claim, do not make that claim
4. Never introduce information not present in the retrieved passages
5. If the evidence is thin or absent for part of the question, say so explicitly

FORMAT RULES:
- Use markdown headers (##, ###) for structure
- Use bullet points or numbered lists for sequential steps
- Use bold for key terms on first use
- Use tables for comparative information
- Write at PhD level — do not oversimplify
- Do not add a preamble like "Great question" — start directly with the Overview section"""

    return prompt


def synthesize_answer(query: str, chunks: list[dict]) -> str:
    """
    Calls Claude Sonnet with maximum tokens for comprehensive synthesis.
    Uses 4096 tokens (vs 1500 before) to allow full exhaustive answers.
    """
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model      = cfg.LLM_MODEL,
        max_tokens = 4096,   # Maximum — allows full comprehensive responses
        messages   = [{"role": "user", "content": build_synthesis_prompt(query, chunks)}],
    )
    return response.content[0].text


# ── Chunk card ────────────────────────────────────────────────────────────────
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
            st.markdown(
                f'[PubMed {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        preview = text[:400] + "..." if len(text) > 400 else text
        st.write(preview)
        if len(text) > 400:
            with st.expander("Show full chunk", expanded=False):
                st.write(text)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(chunks: list[dict] = None):
    with st.sidebar:
        st.markdown("### 🔬 Knowledge Base")
        st.markdown("**Papers:** 3,894")
        st.markdown("**Chunks:** 302,377")
        st.markdown("**Vectors:** Pinecone serverless")
        st.markdown("**Embeddings:** Voyage AI voyage-3")
        st.markdown("**Reranker:** Cohere Rerank 3")
        st.markdown("---")

        st.markdown("### ⚙️ Search Options")
        use_bm25     = st.checkbox("BM25 keyword search", value=True,     key="cb_bm25")
        use_semantic = st.checkbox("Semantic search",     value=True,     key="cb_semantic")
        use_rerank   = st.checkbox("Cohere reranking",    value=True,     key="cb_rerank")
        top_k        = st.slider("Chunks to retrieve", 4, 16, 16, key="slider_topk")
        section_filter = st.multiselect(
            "Filter by section",
            ["methods", "results", "discussion", "abstract", "introduction", "conclusion"],
            default=[],
            help="Leave empty to search all sections",
            key="ms_section",
        )
        st.markdown("---")

        if chunks:
            st.markdown(f"### 📄 Retrieved Chunks ({len(chunks)})")
            for i, chunk in enumerate(chunks, start=1):
                render_chunk_card(chunk, i)

    return use_bm25, use_semantic, use_rerank, top_k, section_filter


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Session state
    for key, default in [
        ("query",          ""),
        ("chunks",         []),
        ("answer",         ""),
        ("timing",         {}),
        ("trigger_search", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Header
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

    use_bm25, use_semantic, use_rerank, top_k, section_filter = render_sidebar(
        st.session_state.chunks or None
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
        if st.button("TTE design",       key="btn_tte"):   set_example("TTE design")
        if st.button("Competing events", key="btn_comp"):  set_example("Competing events")
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
        placeholder      = "e.g. How do I implement sequential trial emulation with time-varying confounding in Medicaid claims data?",
        height           = 80,
        label_visibility = "collapsed",
        key              = "ta_query",
    )

    search_clicked = st.button("🔍 Search & Synthesize", type="primary", key="btn_search")

    should_search = search_clicked or st.session_state.trigger_search
    active_query  = st.session_state.query if st.session_state.trigger_search else user_query

    if should_search and active_query.strip():
        st.session_state.trigger_search = False
        st.session_state.query          = active_query

        pinecone_filter = {"section": {"$in": section_filter}} if section_filter else None

        with st.spinner("Searching 302,377 chunks..."):
            t0 = time.time()

            # ── Query expansion for retrieval ─────────────────────────────
            # Problem: clinically-phrased queries ("gabapentinoid overdose TTE")
            # pull clinical papers (Gomes, Evoy) instead of methods papers
            # (Hernán, Young & Stensrud, Yoshida CCW) because semantic similarity
            # scores drug/outcome terms higher than methodology terms.
            #
            # Fix: prepend a methodology anchor to the retrieval query so
            # the vector search always finds methods papers first.
            # The LLM still receives active_query (original, unchanged) —
            # only the retrieval step uses the expanded version.
            METHODS_ANCHOR = (
                "target trial emulation clone-censor-weight CCW sequential trial "
                "competing events IPCW inverse probability censoring weights "
                "time-varying confounding pharmacoepidemiology causal inference — "
            )
            retrieval_query = METHODS_ANCHOR + active_query

            chunks         = retriever.retrieve(
                query        = retrieval_query,
                top_k        = top_k * 3,
                filters      = pinecone_filter,
                use_bm25     = use_bm25,
                use_semantic = use_semantic,
            )
            retrieval_time = time.time() - t0

        if use_rerank and chunks:
            with st.spinner("Reranking with Cohere..."):
                t1          = time.time()
                # Reranker uses active_query (original) so Cohere scores
                # chunks against what the user actually asked
                chunks      = reranker.rerank(active_query, chunks, top_n=top_k)
                rerank_time = time.time() - t1
        else:
            chunks, rerank_time = chunks[:top_k], 0

        st.session_state.chunks = chunks

        with st.spinner("Synthesizing comprehensive answer with Claude Sonnet..."):
            t2       = time.time()
            answer   = synthesize_answer(active_query, chunks)
            llm_time = time.time() - t2

        st.session_state.answer = answer
        st.session_state.timing = {
            "retrieval": retrieval_time,
            "rerank":    rerank_time,
            "llm":       llm_time,
            "chunks":    len(chunks),
        }
        st.rerun()

    elif should_search:
        st.warning("Please enter a query.")

    # Display answer — native Streamlit markdown for mobile compatibility
    if st.session_state.answer:
        st.markdown("### Answer")
        st.divider()
        # Use st.markdown directly — renders properly on all devices including mobile
        # Headers, bold, tables, bullet points all render correctly
        # No white-on-white issue since Streamlit respects system theme
        st.markdown(st.session_state.answer)
        st.divider()

        t     = st.session_state.timing
        total = t.get("retrieval", 0) + t.get("rerank", 0) + t.get("llm", 0)
        st.caption(
            f"⏱ Retrieval: {t.get('retrieval', 0):.1f}s · "
            f"Rerank: {t.get('rerank', 0):.1f}s · "
            f"LLM: {t.get('llm', 0):.1f}s · "
            f"Total: {total:.1f}s · "
            f"{t.get('chunks', 0)} chunks used · "
            f"Max response: 4,096 tokens"
        )

    # Footer
    st.markdown("---")
    st.caption(
        "REgalys · Real-world Evidence Generation and Analysis Insights · "
        "Built by Ugochukwu Ezigbo BPharm, MHA · "
        "PhD Student, University of Pittsburgh · Pharmaceutical Outcomes & Policy · "
        "[GitHub](https://github.com/Ugogabby/regalys)"
    )


if __name__ == "__main__":
    main()