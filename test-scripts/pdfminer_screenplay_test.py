"""
Test script for the screenplay-aware PDF parser.

Uses the parser from apps/screenplay-parser to parse a screenplay PDF and print
title, authors, scenes, characters, element distribution, and sample content.
Run from repo root or ensure apps/screenplay-parser is on PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from apps/screenplay-parser when run from repo root or test-scripts/
_repo_root = Path(__file__).resolve().parent.parent
_parser_dir = _repo_root / "apps" / "screenplay-parser"
if _parser_dir.is_dir() and str(_parser_dir) not in sys.path:
    sys.path.insert(0, str(_parser_dir))

from screenplay_parser import (  # type: ignore[import-untyped]
    parse_screenplay,
    ScreenplayParseError,
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pdf_path = script_dir / "files" / "marty-supreme-2025.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    # Limit to first 10 pages for faster test runs
    MAX_PAGES = 10
    print(f"Parsing: {pdf_path.name} (first {MAX_PAGES} pages)\n")
    try:
        screenplay = parse_screenplay(pdf_path, max_pages=MAX_PAGES)
    except ScreenplayParseError as e:
        print(f"Parse error: {e}")
        return

    # ── Title page ──────────────────────────────────────────────────────
    print(f"Title:   {screenplay.title}")
    print(f"Authors: {', '.join(screenplay.authors) or '(unknown)'}")
    print(f"Scenes: {len(screenplay.scenes)}")
    print(f"Total elements: {len(screenplay.elements)}")

    # ── Characters ──────────────────────────────────────────────────────
    characters = screenplay.all_characters
    print(f"\nUnique characters ({len(characters)}):")
    for name in sorted(characters):
        print(f"  - {name}")

    # ── Element type distribution ───────────────────────────────────────
    type_counts: dict[str, int] = {}
    for el in screenplay.elements:
        type_counts[el.type.name] = type_counts.get(el.type.name, 0) + 1
    print("\nElement distribution:")
    for name, count in sorted(type_counts.items()):
        print(f"  {name:20s} {count:5d}")

    # ── First 8 scenes ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIRST 8 SCENES")
    print("=" * 70)
    for scene in screenplay.scenes[:8]:
        lt = scene.location_type.value if scene.location_type else "?"
        print(f"\n{'─' * 60}")
        print(f"[Page {scene.page}] {scene.heading}")
        print(f"  Type: {lt}  Location: {scene.location}  Time: {scene.time_of_day}")
        print(f"  Characters: {', '.join(sorted(scene.characters_present)) or '(none)'}")

        for block in scene.dialogue_blocks[:4]:
            parens = f" {block.parentheticals}" if block.parentheticals else ""
            tag = " [V.O.]" if block.is_voice_over else ""
            speech_preview = block.speech[:100]
            if len(block.speech) > 100:
                speech_preview += "..."
            print(f"    {block.character}{tag}{parens}: \"{speech_preview}\"")

        action_preview = " | ".join(scene.action_lines[:3])
        if len(action_preview) > 120:
            action_preview = action_preview[:120] + "..."
        if action_preview:
            print(f"  Action: {action_preview}")

    # ── Sample raw elements (first 40) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("FIRST 40 ELEMENTS (raw)")
    print("=" * 70)
    for i, el in enumerate(screenplay.elements[:40]):
        tag = el.type.name[:12]
        char = f" ({el.character})" if el.character else ""
        text_preview = el.text[:60].replace("\n", "\\n")
        print(f"  {i + 1:3d}. [{tag:12s}]{char} p{el.page:3d} | {text_preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
