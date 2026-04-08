## Source
`school_docs/files/Introduction to Information Retrieval.html`

## Summary
Companion website for the Manning, Raghavan & Schütze textbook *Introduction to Information Retrieval* (Cambridge University Press, 2008). Provides links to the full HTML edition, PDF edition, and Stanford IR lecture slides. The book covers: Boolean retrieval, inverted index construction, tolerant retrieval, scoring and ranking (TF-IDF, BM25), evaluation (Cranfield, MAP, NDCG), probabilistic IR, language models, and web search.

URL: http://nlp.stanford.edu/IR-book/

## Relevance to Implementation
**Indirect — foundational theory reference.** The book is the primary source for:
- TF-IDF weighting (Chapter 6): directly implemented in current vectorizer
- Cosine similarity / VSM (Chapter 6): directly implemented in current scoring
- BM25 (Chapter 11): the assignment requires a BM25 baseline comparison; the textbook is the canonical reference
- Cranfield evaluation / Precision, Recall, MAP (Chapter 8): the assignment requires formal evaluation metrics

No new implementation changes arise from this file alone, but it anchors the theoretical basis for all backlog evaluation items (BL-010).

## Implement Now
No — reference only.

## If Yes — Proposed Change
N/A. Use Chapter 8 (Evaluation) as the specification for implementing Precision@5, Recall@10, MAP, and MRR per BL-010.

## Confidence
High
