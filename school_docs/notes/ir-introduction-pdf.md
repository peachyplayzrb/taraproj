## Source
`school_docs/files/IR Introduction (1).pdf`

## Summary
Lecture slides titled "Information Retrieval in a Nutshell" (72 slides) by Ingo Frommholz (adapted from Baeza-Yates & Ribeiro-Neto and Queen Mary, University of London materials). Table of contents:
1. IR Overview — what IR is, applications (Google, desktop search, enterprise search, RAG with LLMs)
2. Indexing — tokenisation, stopwords, stemming, inverted index
3. Boolean Retrieval Model
4. Vector Space Model (VSM)
5. Term weighting with TF×IDF
6. Further Models — probabilistic IR, language models, **BM25**
7. Conclusion

Key content: VSM represents documents as term-weight vectors; cosine similarity is the standard RSV; TF-IDF is the standard weighting scheme; BM25 improves on TF-IDF with document length normalisation (parameter k1 ~1.2–2.0, b=0.75).

## Relevance to Implementation
**Direct — high relevance for two gaps:**
1. **BM25 baseline** (BL-009): The "Further Models" section covers BM25 with its k1/b parameters. BM25 is the recommended baseline comparison for the assignment. The slides define the BM25 formula and its parameters.
2. **Assignment evidence**: These slides anchor the assignment's expectation that students understand and compare retrieval models (VSM/TF-IDF vs BM25).

No implementation gap with VSM/TF-IDF (already done). Gap is the BM25 comparison.

## Implement Now
No — slides are theory reference. BM25 comparison is tracked in BL-009.

## If Yes — Proposed Change
N/A for slides. For BM25: use Elasticsearch (see `indexing.py` / `retrieve.py`) or `rank-bm25` Python library as a standalone baseline implementation. Compare top-N rankings vs current TF-IDF cosine model on the same query set.

## Confidence
High
