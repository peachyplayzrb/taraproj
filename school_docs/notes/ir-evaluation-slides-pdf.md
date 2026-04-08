## Source
`school_docs/files/IREvaluationSlides.pdf`

## Summary
Lecture slides titled "Evaluating Search Performance" (51 slides) by Ingo Frommholz (BCS Search Solutions 2021 practitioner roundtable). Table of contents:
1. **Evaluation: Why and What?** — motivation (comparing systems, measuring quality, iterative improvement)
2. **The Cranfield Paradigm** — document collection + query set + relevance judgments → offline batch evaluation; reproducible and reusable
3. **Common Evaluation Metrics** — Precision, Recall, F-measure, MAP, NDCG, MRR; P-R curves; evaluation at rank k
4. **Selected Evaluation Initiatives** — TREC, CLEF, NTCIR, MediaEval
5. **Beyond Cranfield** — online evaluation (A/B testing, interleaving), user studies
6. **The Practitioner View** — deployment considerations

Key content: The Cranfield paradigm is the standard offline evaluation method. MAP, NDCG, MRR are the key ranked retrieval metrics. TREC is the canonical evaluation benchmark.

## Relevance to Implementation
**Direct — assignment-critical.** Paired with `IREvaluationExcercise.pdf`, these slides define both the *what* and *why* of the evaluation the assignment requires. Key gaps confirmed:
- No relevance judgments file (qrels) exists yet
- No MAP/NDCG/MRR computation exists yet
- No BM25 comparison baseline exists yet (required for "comparing systems")

The Cranfield paradigm section is particularly important: it defines the minimum artefacts needed (corpus ✓, queries ✗, qrels ✗, metric implementation ✗).

## Implement Now
See `00_admin/combined findings.md`. Implementation tracked in BL-010 (evaluation metrics).

## If Yes — Proposed Change
Same as `IREvaluationExcercise.pdf` note. Minimum Cranfield setup:
1. 5–10 queries representative of CORD-19/arXiv CS topics
2. qrels file: for each query, judge 10–20 CORD-19 docs as relevant/not relevant
3. Evaluation script computing MAP, P@5, Recall@10, MRR, NDCG@10
4. Run against both TF-IDF model (current) and BM25 baseline

## Confidence
High
