---
description: "Use when editing Flask runtime logic in app.py, endpoint handlers, ranking selection, click persistence, or artifact loading behavior."
name: "Python Runtime Rules"
applyTo: "app.py"
---

# Python Runtime Rules

- Preserve current query preprocessing parity with notebook artifacts.
- Keep ranking/sort formulas unchanged unless explicitly requested.
- Maintain artifact compatibility with existing .pkl/.npz files.
- Prefer minimal edits and avoid broad refactors.
- For behavior changes, append one line each to:
  - 00_admin/decision_log.md
  - 00_admin/change_log.md
  - 00_admin/backlog.md
- Verify changed endpoint behavior with at least one direct request path.
