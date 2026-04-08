---
name: notebook-build-check
description: "Use when validating notebook build readiness, artifact serialization completeness, and runtime handoff from notebook to Flask."
---

# Notebook Build Check Skill

Checklist
1. Required notebook outputs exist: vectorizer, pos weights, doc metadata, tfidf matrices.
2. Artifact names match runtime loader expectations.
3. Query preprocessing assumptions remain aligned between notebook and app runtime.

Output
- Missing artifacts
- Naming mismatches
- Suggested minimal fixes
