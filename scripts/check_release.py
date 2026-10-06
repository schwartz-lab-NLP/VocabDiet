"""Check the public source tree for credentials and private machine paths.

Run from any directory: python scripts/check_release.py
Only filenames and finding categories are printed; matched values are never printed.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs", "cache"}
PATTERNS = {
    "embedded Hugging Face credential": re.compile(r"\bhf_" r"[A-Za-z0-9]{20,}\b"),
    "embedded GitHub credential": re.compile(r"\bgh[pousr]_" r"[A-Za-z0-9]{30,}\b"),
    "private home directory": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private cluster directory": re.compile(r"/cs/labs/[^\s\"']+"),
    "interactive debugger": re.compile(r"\b(?:pdb\.set_trace|breakpoint)\s*\("),
    "process-specific tracing": re.compile(r"\bDEBUG_BASE_WORDS(?:_PDB)?\b"),
    "review-process artifact": re.compile(r"\brebuttal\b", re.IGNORECASE),
}


def main():
    findings = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if not path.is_file() or path == Path(__file__).resolve():
            continue
        if path.name in {".env", "AGENTS.md", "CLAUDE.md", "LESSONS.md"}:
            findings.append((relative, "private configuration or process file"))
            continue
        if path.suffix not in {
            ".py",
            ".md",
            ".sh",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
            ".cff",
        }:
            continue
        contents = path.read_text(encoding="utf-8")
        for category, pattern in PATTERNS.items():
            if pattern.search(contents):
                findings.append((relative, category))
    for path, category in findings:
        print(f"{path}: {category}")
    if findings:
        return 1
    print("Public source check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
