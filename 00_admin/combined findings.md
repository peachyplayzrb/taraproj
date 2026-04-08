# Combined Implementation Findings

Aggregated from per-document notes in `school_docs/notes/`. Only actionable implementation improvements are listed here. Theory-only and out-of-scope documents are excluded. Each entry cross-references its source note(s).

---

## Evaluation / Assignment-Critical

### CF-01 — Implement Cranfield evaluation framework (Precision@k, Recall@k, MAP, MRR, NDCG)
| Field | Value |
|---|---|
| Source | [ir-evaluation-slides-pdf.md](ir-evaluation-slides-pdf.md), [ir-evaluation-exercise-pdf.md](ir-evaluation-exercise-pdf.md) |
| Priority | **High** |
| Effort | L |
| Risk if not addressed | Fails assignment evaluation component; no evidence of system quality |

**Proposed change:**
1. Create a relevance judgments file (`qrels.tsv`): minimum 5 queries representative of the CORD-19/arXiv CS domain; for each query, judge 10–20 documents as relevant (1) / not relevant (0).
2. Implement `evaluate.py`: load qrels, run each query against the TF-IDF cosine model, compute per-query: Average Precision, P@5, P@10, Recall@10, MRR, NDCG@10. Average across queries for MAP.
3. Run the same evaluation against the BM25 baseline (CF-02) to produce a comparison table.
4. Report results in the assignment report (table: system × metric).

**Assignment evidence:** The Cranfield paradigm (document corpus ✓, query set ✗, relevance judgments ✗, metric implementation ✗) is the explicit evaluation model taught in the course. MAP, NDCG, and MRR are all explicitly exercised in `IREvaluationExcercise.pdf`. The RAG discussion in Exercise 3b singles out MRR as the appropriate metric when top-1 precision matters.

---

### CF-02 — Implement BM25 baseline for comparison
| Field | Value |
|---|---|
| Source | [ir-introduction-pdf.md](ir-introduction-pdf.md), [cord19-elastic-demo-ipynb.md](cord19-elastic-demo-ipynb.md), [retrieve-py.md](retrieve-py.md) |
| Priority | **High** |
| Effort | M |
| Risk if not addressed | No model comparison; assignment expects baseline vs proposed-method evaluation |

**Proposed change:**
- **Option A (Elasticsearch):** Run `indexing.py` to build the ES `cord19` index, then adapt `retrieve.py` to accept the assignment query set. Capture top-10 ranked docs per query. Compare against TF-IDF cosine results using the qrels from CF-01. Note: `cord_uid` (ES) vs `doi` (current pipeline) field alignment must be verified.
- **Option B (pure Python):** Use `rank-bm25` library (`pip install rank-bm25`) on the existing preprocessed corpus. Easier to integrate with current pipeline; avoids ES dependency.

Option B is recommended for self-contained reproducibility.

---

### CF-03 — Graded relevance judgments for NDCG
| Field | Value |
|---|---|
| Source | [ir-evaluation-exercise-pdf.md](ir-evaluation-exercise-pdf.md) |
| Priority | Medium |
| Effort | S |
| Risk if not addressed | NDCG cannot be computed; binary qrels only support binary metrics |

**Proposed change:** When creating qrels (CF-01), use a 3-level graded scale: `2` = fully relevant, `1` = marginally relevant, `0` = not relevant. This enables NDCG computation without additional tooling changes. Binary-compatible metrics (P@k, MAP, MRR) are unaffected.

---

## Logic / Scoring

### CF-04 — Title-boosting design is confirmed; no change needed
| Field | Value |
|---|---|
| Source | [cord19-queries-ipynb.md](cord19-queries-ipynb.md), [retrieve-py.md](retrieve-py.md) |
| Priority | Low (confirmation, not a gap) |
| Effort | — |
| Risk if not addressed | N/A |

**Note:** The class reference implementation uses `title^3, abstract` field boost (ES `multi_match`). The current TF-IDF implementation achieves equivalent effect via POS-weighted matrix (title terms upweighted). This confirms the design is consistent with course expectations. No change needed.

---

### CF-05 — Score normalisation approach confirmed; no change needed
| Field | Value |
|---|---|
| Source | [retrieve-py.md](retrieve-py.md) |
| Priority | Low (confirmation) |
| Effort | — |
| Risk if not addressed | N/A |

**Note:** `retrieve.py` normalises by `max_score`. The current TF-IDF cosine similarity is inherently normalised by vector norms. Both approaches are consistent. No change needed.

---

## Data / Pipeline

### CF-06 — Verify doc ID alignment for BM25 baseline
| Field | Value |
|---|---|
| Source | [cord19-elastic-demo-ipynb.md](cord19-elastic-demo-ipynb.md), [indexing.py](indexing-py.md) |
| Priority | Medium |
| Effort | S |
| Risk if not addressed | BM25 results cannot be matched against TF-IDF results across systems |

**Proposed change:** When implementing CF-02 Option A (Elasticsearch BM25 baseline), confirm whether the ES index uses `cord_uid` or `doi` as document ID, and map these to the current `doc_ids.pkl` entries. If different, add a lookup table. If using Option B (`rank-bm25`), the existing `doc_ids.pkl` is directly reused and this is a non-issue.

---

## Out of Scope (confirmed not applicable)

The following documents were reviewed and confirmed to have no implementation implications for the current search engine:

| Document | Reason |
|---|---|
| `6CS030 Workshop 2.docx` | Different module; Oracle SQL + Excel data cleaning |
| `6CS030 Workshop 2 Addendum.docx` | Different module; Excel chart compatibility guide |
| `Resident Pay.xlsx` | Workshop 2 dataset; unrelated to IR |
| `Solution_Workshop_2.zip` | Workshop 2 solutions archive; unrelated to IR |
| `WebIRSlides.pdf` | PageRank/HITS require web graph; not applicable |
| `WebIRExercise.pdf` | PageRank/HITS exercises; not applicable |
| `NafBZ_OSN_...html` | OSN behavioral polarity ML paper; no IR relevance |
| `validate_customization.py` | Governance tooling only |
| `Exercise.zip` | Content unknown; no new gaps expected beyond BL-009/BL-010 |
| `L2 Big Data Process.pptx` | Background lecture; no concrete gaps |
| `ScienceDirect_articles_08Apr2026_13-23-10.869.zip` | Possible bibliography sources; extract to confirm |
| `Introduction to Information Retrieval.html` | Textbook companion site; theory reference only |
| `Exercise-1.pdf` | Confirms VSM/TF-IDF already implemented |

---

## Backlog Mapping

| Finding | Existing Backlog Item |
|---|---|
| CF-01 (Cranfield eval) | BL-010 |
| CF-02 (BM25 baseline) | BL-009 |
| CF-03 (graded qrels) | BL-010 (sub-task) |
| CF-04 (title boost confirmed) | N/A — no action |
| CF-05 (score normalisation confirmed) | N/A — no action |
| CF-06 (doc ID alignment) | BL-009 (sub-task) |
