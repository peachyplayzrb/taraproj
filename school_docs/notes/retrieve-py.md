## Source
`school_docs/files/retrieve.py`

## Summary
Elasticsearch query client for the CORD-19 index. Issues a `multi_match` query for `"covid vaccine"` against fields `title^3, abstract` (title boosted 3×). Retrieves results from index `cord19`, normalises scores by dividing each hit's `_score` by `max_score`, then prints normalised score, authors, and title.

Key pattern: score normalisation = `hit['_score'] / result['hits']['max_score']`.

## Relevance to Implementation
**Direct — confirms design decisions:**
1. Field weighting (`title^3 > abstract`) validates the current POS-weighted matrix approach where title terms receive higher weight.
2. Score normalisation by `max_score` is equivalent to the current cosine similarity (already normalised by vector norms). No change needed, but confirms our normalisation direction is correct.
3. The BM25 scoring (ES default) can serve as the BM25 baseline required for the assignment comparison.

## Implement Now
No — the ES retrieval path is a reference showing the class demo approach. The current TF-IDF cosine similarity pipeline is the primary implementation. However, when implementing BM25 baseline (BL-009), `retrieve.py` is the runnable reference to use.

## If Yes — Proposed Change
N/A. For BM25 baseline: run `retrieve.py` with assignment queries against the ES index built by `indexing.py`, capture top-10 results per query, compare Precision@5 / MAP against the TF-IDF cosine model.

## Confidence
High
