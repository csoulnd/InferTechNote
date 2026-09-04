#!/usr/bin/env python3
"""Audit repository Markdown metadata, links, and extraction triggers."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ALLOWED_TYPES = {"moc", "concept", "architecture", "design", "implementation", "investigation", "work", "reference"}
ALLOWED_DOMAINS = {"repository", "agent", "infra", "foundations"}
ALLOWED_STATUSES = {"draft", "active", "evergreen", "archived"}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
ONE_LINE_RE = re.compile(r"^## 一句话解释\s*$\n+([^\n]+)", re.M)
EXTERNAL_LINK_RE = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")


def find_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find repository root")


def parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}, ["missing YAML frontmatter"]
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        return {}, ["unclosed YAML frontmatter"]
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("title", "type", "domain", "status"):
        if not metadata.get(key):
            errors.append(f"missing frontmatter field: {key}")
    return metadata, errors


def local_target(path: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split()[0].strip("<>")
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if target in {"URL", "x"} or "{" in target:
        return None
    target = target.split("#", 1)[0]
    return (path.parent / target).resolve() if target else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="repository path")
    args = parser.parse_args()
    root = find_root(Path(args.path))
    errors: list[str] = []
    warnings: list[str] = []
    stats: Counter[str] = Counter()
    def is_external_checkout(path: Path) -> bool:
        return any((parent / ".git").exists() for parent in path.parents if parent != root)

    markdown = [
        p
        for p in root.rglob("*.md")
        if ".git" not in p.parts and ".agents" not in p.parts and not is_external_checkout(p)
    ]

    for path in sorted(markdown):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            errors.append(f"{rel}: empty Markdown file")
            continue
        metadata, metadata_errors = parse_frontmatter(text)
        errors.extend(f"{rel}: {message}" for message in metadata_errors)
        is_template = rel.startswith("templates/")
        if metadata and not is_template:
            if metadata.get("type") not in ALLOWED_TYPES:
                errors.append(f"{rel}: invalid type {metadata.get('type')!r}")
            if metadata.get("domain") not in ALLOWED_DOMAINS:
                errors.append(f"{rel}: invalid domain {metadata.get('domain')!r}")
            if metadata.get("status") not in ALLOWED_STATUSES:
                errors.append(f"{rel}: invalid status {metadata.get('status')!r}")
            stats[f"type:{metadata.get('type')}"] += 1
            stats[f"status:{metadata.get('status')}"] += 1
            if metadata.get("type") == "concept":
                one_line = ONE_LINE_RE.search(text)
                if one_line is None:
                    errors.append(f"{rel}: concept missing '## 一句话解释'")
                else:
                    sentence_count = len(re.findall(r"[。！？.!?](?:\s|$)", one_line.group(1)))
                    if sentence_count != 1:
                        errors.append(f"{rel}: atomic explanation must be exactly one sentence")
                if not EXTERNAL_LINK_RE.search(text):
                    warnings.append(f"{rel}: concept has no external authoritative reference")
        for raw_target in LINK_RE.findall(text):
            target = local_target(path, raw_target)
            if target is not None and not target.exists():
                errors.append(f"{rel}: broken local link {raw_target!r}")
        in_work = rel.startswith(("agent/", "infra/")) and "/assets/" not in f"/{rel}"
        exempt = path.name == "README.md" or path.name == "learning-roadmap.md"
        if in_work and not exempt and "## Knowledge Extraction" not in text:
            warnings.append(f"{rel}: missing Knowledge Extraction section")
        if in_work:
            stats["work_documents"] += 1
            stats["open_extraction_items"] += len(re.findall(r"^- \[ \].*(?:knowledge|atomic|\u77e5\u8bc6|\u6c89\u6dc0|\u63d0\u53d6)", text, re.M | re.I))

    print(f"Repository: {root}")
    print(f"Markdown files: {len(markdown)}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nAudit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
