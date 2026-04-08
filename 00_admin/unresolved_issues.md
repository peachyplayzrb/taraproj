# Unresolved Issues

Format
- UI-### | Status | Date | Issue | Next Action

Active
- UI-001 | open | 2026-04-08 | Need ongoing discipline to keep governance markdown updated during future behavior changes. | On each behavior change, append one line to decision_log.md and change_log.md.
- UI-002 | open | 2026-04-08 | Optional school_docs category-folder reorganization is deferred by decision to keep school_docs flat for now. | If requested later, run a dedicated non-destructive move pass and append new decision/change log entries.

- UI-003 | open | 2026-04-08 | BL-012 Phase 2 blocked on Colab artifact generation. doc_authors.pkl and doc_author_tokens.pkl must be produced by running the updated notebook in Colab and placed in the repo root before final BL-012 verification and closure. | Run notebook top-to-bottom in Colab, download doc_authors.pkl and doc_author_tokens.pkl from Drive, place both in c:\Users\timsp\search_engine\, re-run smoke_test.py locally, then notify agent to close BL-012.

Resolved
- none
