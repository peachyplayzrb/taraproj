## Source
`school_docs/files/CORD-19 Queries.ipynb`

## Summary
Jupyter notebook demonstrating Elasticsearch querying on a CORD-19 index. Issues a `multi_match` query for `"covid vaccine"` against fields `title^3, abstract` (title boosted 3×) on index `cord19-2`. Prints `max_score` for normalisation, total hit count, and for each hit: normalised score, authors, title.

Two cells: imports/query/print, and an empty/unused commented `match` query block showing the alternative single-field approach.

## Relevance to Implementation
**Direct — validates title-boosting design decision.** The `title^3` field boost in this class reference directly supports the current POS-weighted approach (where title terms are upweighted). This confirms the current implementation's direction is consistent with course expectations. Score normalisation by `max_score` is equivalent to cosine normalisation already in place.

## Implement Now
No — confirms existing design; no change needed.

## If Yes — Proposed Change
N/A. Note: if implementing BM25 baseline, this query structure (`multi_match`, `title^3, abstract`) should be used for parity with the class reference implementation.

## Confidence
High
