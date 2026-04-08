## Source
`school_docs/files/IREvaluationExcercise.pdf`

## Summary
IR Evaluation Exercise Sheet covering:
1. **Cranfield-style evaluation**: what is required (document corpus, query set, relevance judgments, evaluation metric)
2. **Recall and Precision**: given a ranking `r = (+−−+−−−++−)`, compute P@5, P@10, R@5, R@10, average precision; draw the P-R curve
3. **MAP (Mean Average Precision)**: given two systems A and B with 3 queries each, compute MAP for each; determine preferred system; determine which system to use for a RAG approach and which metric is most appropriate for RAG (Precision@1 or MRR)
4. **NDCG and graded relevance**: given graded relevance judgments (0/1/2), compute NDCG@10 for two systems across two queries

The exercise directly specifies the evaluation methodology that the assignment requires to be implemented.

## Relevance to Implementation
**Direct — assignment-critical.** This exercise sheet specifies the exact evaluation methodology the assignment expects:
- Precision@k, Recall@k, Average Precision, MAP → required metrics
- NDCG → graded relevance evaluation
- Cranfield paradigm → need a document corpus + query set + relevance judgments file
- The MAP-vs-RAG discussion → MRR is also a relevant metric to report

These are not yet implemented (backlog BL-010). This is a high-priority gap relative to assignment marks.

## Implement Now
See `00_admin/combined findings.md`. Evaluation is the largest unimplemented assignment component.

## If Yes — Proposed Change
1. Create a relevance judgments file (qrels): minimum 5 queries, each with 10+ documents judged as relevant (1) or not relevant (0).
2. Implement `evaluate.py`: load qrels, run each query against the TF-IDF model and the BM25 baseline, compute P@5, P@10, Recall@10, AP per query, MAP, MRR, NDCG@10.
3. Log results in the notebook or a results table.

## Confidence
High
