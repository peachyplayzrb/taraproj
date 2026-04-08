# Thesis State

Last Updated
- 2026-04-08

Current Objective
- Maintain a method-aware CS paper search engine with transparent ranking and click-aware re-ranking.

Scope Lock
- In scope: workflow control, reproducibility, small safe improvements, verification.
- Out of scope by default: architecture rewrites, framework migrations, major UI redesign.

Canonical Runtime Path
1. Build/update artifacts via notebook path when needed.
2. Serve and test behavior via Flask app runtime path.

Current Priorities
1. Keep behavior stable and explainable.
2. Keep handoff to auto-implementation consistent.
3. Log decisions/changes/issues with lightweight entries.
4. Maintain admin-only requirement traceability alignment to 00_admin/claudeexport and 00_admin/document_pdf.pdf while treating school_docs/files and school_docs/notes as reference-only context; runtime implementation authority remains app.py, templates/index.html, Untitled4 (8).ipynb, and root runtime artifacts.
