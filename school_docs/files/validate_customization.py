from __future__ import annotations

from pathlib import Path
import re
import sys


def find_workspace_root(start: Path) -> Path:
    """Walk up from script location until a .github directory is found."""
    for candidate in [start, *start.parents]:
        if (candidate / ".github").exists():
            return candidate
    # Fallback for unexpected layouts; preserves previous behavior.
    return start.parents[1]


ROOT = find_workspace_root(Path(__file__).resolve().parent)
GITHUB = ROOT / ".github"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") and "\n---\n" in text[4:]


def frontmatter_block(text: str) -> str:
    end = text.find("\n---\n", 4)
    if end == -1:
        return ""
    return text[4:end]


def kv_pairs(block: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        pairs[k.strip()] = v.strip().strip('"')
    return pairs


def check_markdown_file(path: Path, require_name: bool = False) -> list[str]:
    errors: list[str] = []
    text = read_text(path)
    if not has_frontmatter(text):
        errors.append(f"{path}: missing YAML frontmatter")
        return errors
    pairs = kv_pairs(frontmatter_block(text))
    if "description" not in pairs or not pairs["description"]:
        errors.append(f"{path}: missing required frontmatter field: description")
    if require_name and ("name" not in pairs or not pairs["name"]):
        errors.append(f"{path}: missing required frontmatter field: name")
    return errors


def main() -> int:
    errors: list[str] = []

    for p in (GITHUB / "agents").glob("*.agent.md"):
        errors.extend(check_markdown_file(p, require_name=True))

    for p in (GITHUB / "instructions").glob("*.instructions.md"):
        errors.extend(check_markdown_file(p, require_name=False))

    for p in (GITHUB / "prompts").glob("*.prompt.md"):
        errors.extend(check_markdown_file(p, require_name=False))

    for p in (GITHUB / "skills").glob("*/SKILL.md"):
        errors.extend(check_markdown_file(p, require_name=True))

    hooks_path = GITHUB / "hooks" / "hooks.json"
    if not hooks_path.exists():
        errors.append(f"{hooks_path}: missing hooks config")
    else:
        text = read_text(hooks_path)
        if '"hooks"' not in text:
            errors.append(f"{hooks_path}: missing top-level hooks key")

    if errors:
        print("Customization validation failed:\n")
        for e in errors:
            print(f"- {e}")
        return 1

    print("Customization validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
