---
description: "Use when editing notebook cells, ETL flow, vectorization steps, serialization outputs, or Colab run instructions for .ipynb files."
name: "Notebook Pipeline Rules"
applyTo: "**/*.ipynb"
---

# Notebook Pipeline Rules

- Keep cell order stable unless explicit reordering is requested.
- Preserve serialized artifact names and formats expected by runtime app.
- Do not change ranking methodology by default.
- Avoid replacing existing pipeline with a different framework.
- Keep edits local to requested cells and document assumptions in markdown cells only if requested.
- Verify notebook snippets for variable dependencies when adding/reordering cells.
