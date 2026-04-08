## Source
`school_docs/files/Exercise-1.pdf`

## Summary
IR Exercise Sheet 1 — foundational exercises covering:
1. **IR vs DB**: differences between information retrieval and databases; exact vs best match
2. **Conceptual model and system architecture**: describe IR conceptual model and high-level software architecture
3. **Indexing**: 4-step document indexing process (tokenisation, stopword removal, stemming, inverted index construction)
4. **Inverted list**: role and advantages of inverted lists for indexing and retrieval
5. **Boolean retrieval with inverted lists**: processing `cat OR dog` via inverted lists and Boolean model
6. **RSVs and the Vector Space Model**: given 3 document vectors and a query vector, compute RSV using scalar product; derive ranking
7. **TF-IDF and VSM**: given 3 documents expressed as term sets (after stopword removal), construct a TF-IDF weighted VSM; compute cosine similarity ranking

The exercise directly maps to the theoretical foundations of the current implementation (TF-IDF, VSM, cosine similarity).

## Relevance to Implementation
**Indirect — confirms theoretical basis.** The TF-IDF × cosine similarity approach exercised here is exactly what the current implementation uses. No gap with current implementation from this exercise sheet; it confirms the assignment expects VSM as the core retrieval model.

## Implement Now
No — theory reference only; implementation already in place.

## If Yes — Proposed Change
N/A.

## Confidence
High
