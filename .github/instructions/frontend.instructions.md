---
description: "Use when editing search UI templates, result cards, sorting controls, or click-handling scripts in HTML files under templates/."
name: "Frontend Template Rules"
applyTo: "templates/**/*.html"
---

# Frontend Template Rules

- Preserve existing UX structure unless redesign is explicitly requested.
- Keep click recording and result rendering behavior consistent with backend payloads.
- Avoid introducing framework migrations.
- Keep JavaScript changes small, deterministic, and easy to debug.
- Prefer adding clear labels for score fields over changing score semantics.
