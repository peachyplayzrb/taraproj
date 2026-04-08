# Unresolved Issues

Format
- UI-### | Status | Date | Issue | Next Action

Active
- UI-001 | open | 2026-04-08 | Need ongoing discipline to keep governance markdown updated during future behavior changes. | On each behavior change, append one line to decision_log.md and change_log.md.
- UI-002 | open | 2026-04-08 | Optional school_docs category-folder reorganization is deferred by decision to keep school_docs flat for now. | If requested later, run a dedicated non-destructive move pass and append new decision/change log entries.

Resolved
- UI-003 | resolved | 2026-04-09 | BL-012 Phase 2 artifact-generation blocker cleared. doc_authors.pkl and doc_author_tokens.pkl are present in repo root, and local smoke_test.py confirms author artifacts load and align. | Keep author artifact regeneration in normal notebook runbook when rebuilding index artifacts.
