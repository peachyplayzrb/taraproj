## Source
`school_docs/files/CORD-19 Elastic Demo.ipynb`

## Summary
Jupyter notebook demonstrating indexing of CORD-19 abstracts into Elasticsearch. Reads `metadata.csv`, creates an index named `cord19`, and indexes each row with fields `authors`, `title`, `abstract` using `cord_uid` as the document ID. References:
- Elasticsearch Python API (v7.17)
- CORD-19 dataset (Allen AI)
- BM25 algorithm explanation (Elastic blog)

Three cells: imports/index-creation/CSV loop, count print, and an empty cell.

## Relevance to Implementation
**Indirect — confirms BM25 baseline path.** The notebook was likely the in-class starting point before the assignment extended it to TF-IDF + POS weighting. The ES index built here uses BM25 by default (Okapi BM25 is Elasticsearch's default similarity). This notebook, combined with `retrieve.py`, constitutes a runnable BM25 baseline for assignment comparison.

The `cord_uid` ID field differs from the current implementation's `doi`-based `doc_ids.pkl` — check field alignment when running the BM25 baseline experiment.

## Implement Now
No — the notebook is reference context. For BM25 baseline implementation, see backlog BL-009.

## If Yes — Proposed Change
N/A. When implementing BM25 baseline: run this notebook to build the ES index, then adapt `retrieve.py` with assignment queries and capture ranked lists for metric calculation.

## Confidence
High
