"""
pico_extractor.py
──────────────────
REgalys PICO Extraction & Study Quality Assessment

Two capabilities in one module:

1. PICO Extraction
   Structured extraction of Population, Intervention, Comparator, Outcome
   from retrieved pharmacoepidemiology chunks using Claude Haiku.
   Outputs are directly compatible with HTA evidence table formats
   (NICE, ICER, G-BA dossier submissions).

2. Study Quality Assessment
   Automated bias flag detection per chunk. Checks for:
   - Active comparator design (vs non-user comparator)
   - New user design (vs prevalent user)
   - Validated outcome definition
   - Confounding adjustment method (PS, IPTW, multivariable, none)
   - Competing events handling
   - Immortal time bias protection

These two capabilities run together on each retrieved chunk in a single
Claude Haiku call, keeping latency and cost minimal (~$0.001 per chunk).

Integration:
   from pico_extractor import PICOEnricher
   enricher = PICOEnricher()
   enriched_chunks = enricher.enrich(chunks)         # batch
   evidence_table  = enricher.to_evidence_table(enriched_chunks)
   quality_summary = enricher.quality_summary(enriched_chunks)
"""

import json
import anthropic
from typing import Optional
from dataclasses import dataclass, asdict, field

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PICOElements:
    """Structured PICO extraction output."""
    population:    str   # patient group, setting, age, comorbidities
    intervention:  str   # drug/exposure of interest
    comparator:    str   # active comparator, control, or "Not reported"
    outcome:       str   # primary endpoint(s)
    study_design:  str   # TTE, cohort, case-control, RCT, etc.
    time_horizon:  str   # follow-up duration if mentioned
    data_source:   str   # claims, EHR, registry, trial database
    confidence:    str   # high | medium | low


@dataclass
class QualityFlags:
    """
    Automated study quality flags for pharmacoepidemiology.
    Each flag is True / False / None (not determinable from text).
    """
    active_comparator:        Optional[bool]   # used an active comparator (not non-users)
    new_user_design:          Optional[bool]   # restricted to new users / incident users
    validated_outcome:        Optional[bool]   # outcome validated or from published algorithm
    confounding_method:       str              # IPTW | PS matching | regression | none | unclear
    competing_events_handled: Optional[bool]   # explicitly addressed competing events
    immortal_time_protected:  Optional[bool]   # explicitly protected against immortal time bias
    overall_quality:          str              # strong | moderate | weak | not determinable


@dataclass
class EnrichedChunk:
    """
    A retrieved chunk augmented with PICO and quality assessments.
    Wraps the original chunk dict so all original fields are preserved.
    """
    original:       dict
    pico:           Optional[PICOElements]
    quality:        Optional[QualityFlags]
    extraction_ok:  bool = True

    # Passthrough helpers for common fields
    @property
    def title(self):   return self.original.get("title", "Unknown")
    @property
    def pmid(self):    return self.original.get("pmid", "")
    @property
    def year(self):    return self.original.get("year", "")
    @property
    def authors(self): return self.original.get("authors", "")
    @property
    def journal(self): return self.original.get("journal", "")
    @property
    def text(self):
        return self.original.get("text_original", self.original.get("text", ""))

    def to_evidence_table_row(self) -> Optional[dict]:
        """Format as HTA-style evidence table row. Returns None if PICO extraction failed."""
        if not self.pico:
            return None
        return {
            "Title":                    self.title[:120],
            "PMID":                     self.pmid,
            "Year":                     self.year,
            "Study Design":             self.pico.study_design,
            "Data Source":              self.pico.data_source,
            "Population":               self.pico.population,
            "Intervention":             self.pico.intervention,
            "Comparator":               self.pico.comparator,
            "Outcome":                  self.pico.outcome,
            "Time Horizon":             self.pico.time_horizon,
            "PICO Confidence":          self.pico.confidence,
            # Quality flags
            "Active Comparator":        _flag_str(self.quality.active_comparator)        if self.quality else "—",
            "New User Design":          _flag_str(self.quality.new_user_design)          if self.quality else "—",
            "Validated Outcome":        _flag_str(self.quality.validated_outcome)        if self.quality else "—",
            "Confounding Method":       self.quality.confounding_method                  if self.quality else "—",
            "Competing Events":         _flag_str(self.quality.competing_events_handled) if self.quality else "—",
            "Immortal Time Protected":  _flag_str(self.quality.immortal_time_protected)  if self.quality else "—",
            "Overall Quality":          self.quality.overall_quality                     if self.quality else "—",
        }


def _flag_str(val: Optional[bool]) -> str:
    """Converts a bool/None quality flag to a display string."""
    if val is True:   return "✓ Yes"
    if val is False:  return "✗ No"
    return "?"


# ─────────────────────────────────────────────────────────────────────────────
# Combined extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

_COMBINED_SYSTEM = """You are an expert pharmacoepidemiologist and systematic literature 
reviewer with deep expertise in HEOR, HTA submissions, and real-world evidence methods.

You will receive a text excerpt from a pharmacoepidemiology paper. Return ONLY valid JSON 
with two top-level keys: "pico" and "quality". No preamble. No markdown. No explanation.

=== PICO EXTRACTION ===
Extract:
  population:    Patient group (age, diagnosis, setting, comorbidities). 1-3 sentences.
  intervention:  The drug, treatment, or exposure of interest. Be specific (drug name, dose if given).
  comparator:    The comparison group. If active comparator, name the drug. Write "Not reported" if absent.
  outcome:       Primary endpoint(s). Include clinical, safety, economic if mentioned.
  study_design:  Design type. Examples: "New-user active comparator cohort", 
                 "Target trial emulation", "Self-controlled case series", "RCT", etc.
  time_horizon:  Follow-up duration. Write "Not reported" if absent.
  data_source:   Data source type. Examples: "Medicare claims", "EHR", "T-MSIS Medicaid", 
                 "linked registry-claims", "trial database". Write "Not reported" if absent.
  confidence:    "high" if all 4 core PICO elements explicit, "medium" if 2-3 explicit, 
                 "low" if mostly absent or inferred.

=== QUALITY ASSESSMENT ===
active_comparator:        true if study explicitly uses an active drug comparator (not non-users/no treatment).
                          false if compared to non-users or general population. null if unclear.
new_user_design:          true if restricted to incident/new users with a washout period.
                          false if prevalent users or unclear inclusion. null if not stated.
validated_outcome:        true if outcome definition is cited from a validation study or uses 
                          published algorithms with sensitivity/specificity reported.
                          false if outcome is not validated. null if unclear.
confounding_method:       One of: "IPTW", "PS matching", "multivariable regression", 
                          "PS + regression", "TMLE", "none", "unclear".
competing_events_handled: true if competing events (e.g. death competing with non-fatal outcome) 
                          are explicitly addressed analytically. false if ignored. null if unclear.
immortal_time_protected:  true if the study explicitly protects against immortal time bias 
                          (e.g. new user design, time-conditional PS, landmark analysis).
                          false if design has apparent immortal time issue. null if unclear.
overall_quality:          "strong"   = active comparator + new user + validated outcome + adequate confounding
                          "moderate" = 2-3 of the above criteria met
                          "weak"     = 0-1 criteria met
                          "not determinable" = insufficient information

Output format — EXACTLY this JSON, nothing else:
{
  "pico": {
    "population": "...",
    "intervention": "...",
    "comparator": "...",
    "outcome": "...",
    "study_design": "...",
    "time_horizon": "...",
    "data_source": "...",
    "confidence": "high|medium|low"
  },
  "quality": {
    "active_comparator": true|false|null,
    "new_user_design": true|false|null,
    "validated_outcome": true|false|null,
    "confounding_method": "...",
    "competing_events_handled": true|false|null,
    "immortal_time_protected": true|false|null,
    "overall_quality": "strong|moderate|weak|not determinable"
  }
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Main enricher class
# ─────────────────────────────────────────────────────────────────────────────

class PICOEnricher:
    """
    Enriches retrieved RAG chunks with PICO elements and study quality flags.

    Uses Claude Haiku for speed and cost efficiency — one API call per chunk
    returns both PICO and quality in a single JSON response.

    Usage:
        enricher = PICOEnricher()

        # Enrich a list of chunks from HybridRetriever
        enriched = enricher.enrich(chunks)

        # Get HTA-ready evidence table
        table = enricher.to_evidence_table(enriched)

        # Get quality summary across all retrieved papers
        summary = enricher.quality_summary(enriched)
    """

    def __init__(
        self,
        model:       str   = "claude-haiku-4-5-20251001",
        max_tokens:  int   = 700,
        temperature: float = 0.0,
    ):
        self.client      = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

    def _extract_one(self, text: str) -> tuple[Optional[PICOElements], Optional[QualityFlags]]:
        """
        Runs a single Claude Haiku call to extract PICO + quality from one chunk.

        Returns (PICOElements, QualityFlags) or (None, None) on failure.
        """
        try:
            response = self.client.messages.create(
                model       = self.model,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                system      = _COMBINED_SYSTEM,
                messages    = [{
                    "role":    "user",
                    "content": f"Extract PICO and quality assessment from this text:\n\n{text[:2000]}"
                }],
            )

            raw = response.content[0].text.strip()

            # Defensive stripping of markdown fences despite prompt instructions
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            # ── Parse PICO ────────────────────────────────────────────────────
            p = parsed.get("pico", {})
            pico = PICOElements(
                population   = p.get("population",   "Not reported"),
                intervention = p.get("intervention", "Not reported"),
                comparator   = p.get("comparator",   "Not reported"),
                outcome      = p.get("outcome",      "Not reported"),
                study_design = p.get("study_design", "Not reported"),
                time_horizon = p.get("time_horizon", "Not reported"),
                data_source  = p.get("data_source",  "Not reported"),
                confidence   = p.get("confidence",   "low"),
            )

            # ── Parse quality ─────────────────────────────────────────────────
            q = parsed.get("quality", {})
            quality = QualityFlags(
                active_comparator        = q.get("active_comparator"),
                new_user_design          = q.get("new_user_design"),
                validated_outcome        = q.get("validated_outcome"),
                confounding_method       = q.get("confounding_method", "unclear"),
                competing_events_handled = q.get("competing_events_handled"),
                immortal_time_protected  = q.get("immortal_time_protected"),
                overall_quality          = q.get("overall_quality", "not determinable"),
            )

            return pico, quality

        except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
            print(f"  [PICOEnricher] Extraction failed: {e}")
            return None, None

    def enrich(self, chunks: list[dict], verbose: bool = False) -> list[EnrichedChunk]:
        """
        Batch enriches a list of retrieved chunks.

        Args:
            chunks:  list of chunk dicts from HybridRetriever.retrieve()
            verbose: print progress per chunk

        Returns:
            list of EnrichedChunk objects preserving all original metadata
        """
        enriched = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text_original", chunk.get("text", ""))

            if not text.strip():
                enriched.append(EnrichedChunk(
                    original=chunk, pico=None, quality=None, extraction_ok=False
                ))
                continue

            if verbose:
                authors = chunk.get("authors", "Unknown")[:40]
                year    = chunk.get("year", "")
                print(f"  [{i+1}/{len(chunks)}] Extracting: {authors} ({year})")

            pico, quality = self._extract_one(text)

            enriched.append(EnrichedChunk(
                original      = chunk,
                pico          = pico,
                quality       = quality,
                extraction_ok = (pico is not None),
            ))

        return enriched

    def to_evidence_table(
        self,
        enriched:    list[EnrichedChunk],
        min_confidence: str = "medium",   # filter out low-confidence extractions
    ) -> list[dict]:
        """
        Returns an HTA-style evidence table from enriched chunks.

        Args:
            enriched:       output of enrich()
            min_confidence: "high" | "medium" | "low" — filter threshold

        Returns:
            list of evidence table row dicts ready for st.dataframe() or CSV export
        """
        confidence_rank = {"high": 2, "medium": 1, "low": 0}
        min_rank = confidence_rank.get(min_confidence, 1)

        table = []
        for ec in enriched:
            if not ec.extraction_ok or not ec.pico:
                continue
            if confidence_rank.get(ec.pico.confidence, 0) < min_rank:
                continue
            row = ec.to_evidence_table_row()
            if row:
                table.append(row)

        return table

    def quality_summary(self, enriched: list[EnrichedChunk]) -> dict:
        """
        Aggregate quality statistics across all enriched chunks.

        Returns a dict suitable for display in the Streamlit sidebar.
        """
        total = len([e for e in enriched if e.quality])
        if total == 0:
            return {"total_assessed": 0}

        def pct_true(attr: str) -> str:
            vals = [getattr(e.quality, attr) for e in enriched if e.quality]
            yes  = sum(1 for v in vals if v is True)
            return f"{yes}/{total} ({100*yes//total}%)"

        quality_counts = {"strong": 0, "moderate": 0, "weak": 0, "not determinable": 0}
        for e in enriched:
            if e.quality:
                quality_counts[e.quality.overall_quality] = \
                    quality_counts.get(e.quality.overall_quality, 0) + 1

        confounding_counts: dict[str, int] = {}
        for e in enriched:
            if e.quality:
                m = e.quality.confounding_method
                confounding_counts[m] = confounding_counts.get(m, 0) + 1

        return {
            "total_assessed":            total,
            "active_comparator":         pct_true("active_comparator"),
            "new_user_design":           pct_true("new_user_design"),
            "validated_outcome":         pct_true("validated_outcome"),
            "competing_events_handled":  pct_true("competing_events_handled"),
            "immortal_time_protected":   pct_true("immortal_time_protected"),
            "quality_distribution":      quality_counts,
            "confounding_methods":       confounding_counts,
        }
