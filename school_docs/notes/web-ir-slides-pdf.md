## Source
`school_docs/files/WebIRSlides.pdf`

## Summary
Lecture slides titled "Web Information Retrieval" (46 slides) by Ingo Frommholz. Table of contents:
1. **Seeking Information on the Web** — scale, diversity, link structure
2. **Anatomy of a Web Search Engine (early Google)** — crawler, index, query processor, ranker; references Brin & Page 1998
3. **Link Analysis with PageRank and HITS** — PageRank formula (damping factor β), iterative computation, HITS (hub/authority scores), Kleinberg 1998
4. **Conclusion**
5. **References** — Baeza-Yates & Ribeiro-Neto (2011), Manning et al. (2008), Brin & Page (1998), Page et al. (1998), Kleinberg (1998)

## Relevance to Implementation
**None — out of scope.** Link analysis (PageRank, HITS) requires a hyperlink graph. The current implementation operates on a static flat document corpus with no inter-document links. The "anatomy of a web search engine" section is educational background confirming the current system is a simplified subset of a full web search engine.

## Implement Now
No.

## If Yes — Proposed Change
N/A.

## Confidence
High
