## Source
`school_docs/files/indexing.py`

## Summary
Elasticsearch CORD-19 indexer. Reads `metadata.csv` from the CORD-19 dataset and bulk-indexes documents into an ES index named `cord19_2020-03-20`. Each document is keyed by `doi`, with fields: `title`, `abstract`, `authors`. Uses the `elasticsearch` Python client against `http://localhost:9200`. Prints a progress counter every 10,000 documents.

Key fields indexed: `title`, `abstract`, `authors` (split on `; `). Document ID is `doi`.

## Relevance to Implementation
**Indirect.** The field structure (`title`, `abstract`, `doi`) maps directly onto the current TF-IDF pipeline's `doc_ids.pkl`, `doc_titles.pkl`, `doc_abstracts.pkl` artifacts. The field choices (title + abstract) are consistent with the current vectorization approach. The ES index uses BM25 by default (Elasticsearch default scoring), which is the intended BM25 baseline for assignment comparison.

## Implement Now
No — the current pipeline is TF-IDF based and does not use Elasticsearch. The indexer is reference context only.

## If Yes — Proposed Change
N/A. If a BM25 baseline is needed, run `indexing.py` against a local ES instance and compare results with `retrieve.py`. This is already tracked in backlog BL-009/BL-013.

## Confidence
High
