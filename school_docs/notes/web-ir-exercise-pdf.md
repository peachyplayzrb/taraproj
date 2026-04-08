## Source
`school_docs/files/WebIRExercise.pdf`

## Summary
Web Information Retrieval Exercise Sheet covering:
1. **Anatomy of a web search engine**: crawler, indexer, ranker, query processor, UI — describe each component and relationships
2. **PageRank and the Random Surfer Model**: describe how iterative PageRank formula implements the random surfer; given a hypertext graph (d1–d4), compute PageRank with β=0.8 after 2 iterations; compute hub and authority values via HITS after 2 iterations
3. **Link analysis 2**: further PageRank computation on a different graph; compute convergence behaviour
4. **Anchor text**: how anchor text improves retrieval quality; when is it misleading?

The exercises build on the WebIR slides and Manning et al. textbook.

## Relevance to Implementation
**None — out of scope.** The current implementation is a document IR system over a static arXiv/CORD-19 corpus, with no hyperlink graph. PageRank and HITS require a web graph and are not applicable. Anchor text is not applicable to abstract-only documents.

## Implement Now
No.

## If Yes — Proposed Change
N/A.

## Confidence
High
