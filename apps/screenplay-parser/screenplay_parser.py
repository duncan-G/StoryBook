"""
Screenplay-aware PDF parser.

Architected for long-term maintainability. Separates PDF extraction (I/O),
layout classification (Heuristics), and document assembly (State/Builder)
into distinct, easily testable components.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Iterator

try:
    from opentelemetry import trace
    _tracer = trace.get_tracer(__name__, "0.1.0")
except ImportError:
    trace = None  # type: ignore[assignment]
    _tracer = None


@contextmanager
def _null_span():
    """No-op context manager when OpenTelemetry is not available."""
    yield


# ---------------------------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenplayConfig:
    """Holds all heuristic rules, keywords, and default thresholds."""

    default_action_x: float = 108.0
    default_dialogue_x: float = 180.0
    default_paren_x: float = 209.0
    default_char_x: float = 252.0

    x_page_number: float = 490.0
    x_transition: float = 430.0

    time_of_day_keywords: frozenset[str] = frozenset({
        "DAY", "NIGHT", "DAWN", "DUSK", "SUNSET", "SUNRISE",
        "EVENING", "MORNING", "AFTERNOON", "LATER", "CONTINUOUS",
        "MOMENTS LATER", "SAME TIME", "MAGIC HOUR",
    })
    transition_keywords: frozenset[str] = frozenset({
        "CUT TO", "FADE OUT", "FADE IN", "DISSOLVE TO",
        "SMASH CUT", "SLAM CUT", "HARD CUT",
    })
    section_keywords: frozenset[str] = frozenset({
        "PROLOGUE", "EPILOGUE", "THE END", "END CREDITS", "TITLE CARD",
    })
    continuation_markers: frozenset[str] = frozenset({
        "(MORE)", "(CONTINUED)", "CONTINUED:",
    })
    credit_keywords: frozenset[str] = frozenset({
        "written by", "screenplay by", "story by", "teleplay by",
        "screen story by", "based on", "created by",
    })

    scene_heading_pattern: re.Pattern = field(  # type: ignore[type-arg]
        default=re.compile(
            r"^(INT\.|EXT\.|INT\./EXT\.?|I/E\.?|INT/EXT\.?)\s+", re.IGNORECASE,
        ),
        repr=False,
    )

    location_type_map: dict[str, str] = field(default_factory=lambda: {
        "INT.": "INT",
        "EXT.": "EXT",
        "INT./EXT.": "I/E",
        "INT/EXT.": "I/E",
        "I/E.": "I/E",
    })


# ---------------------------------------------------------------------------
# 2. Pure Domain Models
# ---------------------------------------------------------------------------

class ElementType(Enum):
    SCENE_HEADING = auto()
    ACTION = auto()
    DIALOGUE = auto()
    VOICE_OVER = auto()
    PARENTHETICAL = auto()
    TRANSITION = auto()
    SECTION_HEADER = auto()


class InternalType(Enum):
    """Types used only during the pipeline, never in final output."""
    CHARACTER = auto()
    PAGE_NUMBER = auto()


class LocationType(Enum):
    INTERIOR = "INT"
    EXTERIOR = "EXT"
    INTERIOR_EXTERIOR = "I/E"


_LOCATION_TYPE_MAP: dict[str, LocationType] = {
    "INT.": LocationType.INTERIOR,
    "EXT.": LocationType.EXTERIOR,
    "INT./EXT.": LocationType.INTERIOR_EXTERIOR,
    "INT/EXT.": LocationType.INTERIOR_EXTERIOR,
    "I/E.": LocationType.INTERIOR_EXTERIOR,
}


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def merge(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


@dataclass
class SceneElement:
    type: ElementType
    text: str
    page: int
    bbox: BoundingBox
    character: str | None = None


@dataclass
class DialogueBlock:
    character: str
    is_voice_over: bool = False
    lines: list[SceneElement] = field(default_factory=list)

    @property
    def speech(self) -> str:
        return " ".join(
            el.text for el in self.lines
            if el.type in (ElementType.DIALOGUE, ElementType.VOICE_OVER)
        )

    @property
    def parentheticals(self) -> list[str]:
        return [el.text for el in self.lines if el.type == ElementType.PARENTHETICAL]


@dataclass
class Scene:
    heading: str
    page: int
    location_type: LocationType | None = None
    location: str = ""
    time_of_day: str = ""
    elements: list[SceneElement] = field(default_factory=list)

    @property
    def dialogue_blocks(self) -> list[DialogueBlock]:
        blocks: list[DialogueBlock] = []
        current: DialogueBlock | None = None
        for el in self.elements:
            if el.type in (ElementType.DIALOGUE, ElementType.VOICE_OVER, ElementType.PARENTHETICAL):
                char = el.character or ""
                if current is None or current.character != char:
                    current = DialogueBlock(
                        character=char,
                        is_voice_over=(el.type == ElementType.VOICE_OVER),
                    )
                    blocks.append(current)
                current.lines.append(el)
            else:
                current = None
        return blocks

    @property
    def action_lines(self) -> list[str]:
        return [el.text for el in self.elements if el.type == ElementType.ACTION]

    @property
    def characters_present(self) -> set[str]:
        return {el.character for el in self.elements if el.character is not None}


@dataclass
class Screenplay:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    elements: list[SceneElement] = field(default_factory=list)

    @property
    def all_characters(self) -> set[str]:
        return {el.character for el in self.elements if el.character}


# ---------------------------------------------------------------------------
# 3. I/O Boundary (PDF Extraction)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawLine:
    """The boundary between PDF extraction and our business logic."""
    page_num: int
    text: str
    bbox: BoundingBox


_SPURIOUS_SPACE_PREFIX = re.compile(
    r"^(IN|EX)\s+(T\.|T\./EX\s*T\.?)",
    re.IGNORECASE,
)


def _normalize_pdf_text(text: str) -> str:
    """Collapse spurious spaces introduced by PDFMiner in Form XObject PDFs.

    Some PDFs (e.g. Final Draft exports) have slightly enlarged character gaps
    that cause PDFMiner to insert word-break spaces inside tokens like ``INT.``
    or ``EXT.``, producing ``IN T.`` or ``EX T.``.
    """
    m = _SPURIOUS_SPACE_PREFIX.match(text)
    if m:
        fixed_prefix = m.group(0).replace(" ", "")
        text = fixed_prefix + text[m.end():]
    return text


def extract_pdf_lines(pdf_path: Path, max_pages: int | None = None) -> Iterator[RawLine]:
    """
    Isolates third-party PDF library dependency.
    Yields raw lines in reading order.
    If max_pages is set, only the first max_pages pages are extracted.
    """
    import logging

    from pdfminer.high_level import extract_pages
    from pdfminer.layout import (
        LAParams,
        LTContainer,
        LTTextBoxHorizontal,
        LTTextLineHorizontal,
    )

    # Suppress pdfminer debug logs (e.g. psparser) which are extremely verbose per page
    for name in ("pdfminer", "pdfminer.psparser"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # all_texts=True forces layout analysis inside LTFigure containers,
    # which is required for PDFs exported from Final Draft and similar tools
    # that wrap every page in a Form XObject.
    params = LAParams(line_margin=0.3, all_texts=True)
    raw_lines: list[RawLine] = []
    kwargs: dict = {"laparams": params}
    if max_pages is not None and max_pages > 0:
        kwargs["maxpages"] = max_pages

    def _collect_lines(container: LTContainer, page_num: int) -> None:
        for element in container:
            if isinstance(element, LTTextBoxHorizontal):
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        text = _normalize_pdf_text(line.get_text().strip())
                        if text:
                            x0, y0, x1, y1 = line.bbox
                            bbox = BoundingBox(x0, y0, x1, y1)
                            raw_lines.append(RawLine(page_num, text, bbox))
            elif isinstance(element, LTContainer):
                _collect_lines(element, page_num)

    ctx = _tracer.start_as_current_span("screenplay.extract_pdf_lines") if _tracer else _null_span()
    with ctx:
        span = trace.get_current_span() if (trace is not None and _tracer) else None
        if span is not None and span.is_recording():
            span.set_attribute("screenplay.pdf_path", str(pdf_path))
        for page_num, page_layout in enumerate(extract_pages(str(pdf_path), **kwargs), start=1):
            _collect_lines(page_layout, page_num)
        if span is not None and span.is_recording():
            span.set_attribute("screenplay.raw_line_count", len(raw_lines))
            span.set_attribute("screenplay.page_count", max((r.page_num for r in raw_lines), default=0))

    raw_lines.sort(key=lambda r: (r.page_num, -r.bbox.y0, r.bbox.x0))
    yield from raw_lines


# ---------------------------------------------------------------------------
# 4. Analysis & Heuristics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndentationThresholds:
    action_max: float
    dialogue_min: float
    dialogue_max: float
    paren_min: float
    paren_max: float
    character_min: float
    character_max: float


class LayoutAnalyzer:
    """Analyzes spatial layout to determine element types."""

    def __init__(self, config: ScreenplayConfig):
        self.config = config

    def detect_thresholds(self, raw_lines: Iterable[RawLine]) -> IndentationThresholds:
        hist: Counter[int] = Counter()
        for line in raw_lines:
            if line.page_num == 1 or line.bbox.x0 >= self.config.x_transition:
                continue
            hist[round(line.bbox.x0)] += 1

        if not hist:
            return self._build_thresholds(
                self.config.default_action_x, self.config.default_dialogue_x,
                self.config.default_paren_x, self.config.default_char_x,
            )

        clusters = self._cluster_histogram(hist)

        if len(clusters) >= 4:
            ax, dx, px, cx = self._match_clusters_to_roles(clusters)
        elif len(clusters) == 3:
            ax, dx, cx = clusters[0], clusters[1], clusters[2]
            px = (dx + cx) / 2
        else:
            ax, dx = self.config.default_action_x, self.config.default_dialogue_x
            px, cx = self.config.default_paren_x, self.config.default_char_x

        return self._build_thresholds(ax, dx, px, cx)

    def classify_line(self, line: RawLine, th: IndentationThresholds) -> ElementType | InternalType:
        if line.page_num == 1:
            return ElementType.SECTION_HEADER

        x0 = line.bbox.x0
        if x0 >= self.config.x_page_number:
            return InternalType.PAGE_NUMBER
        if x0 >= self.config.x_transition:
            return ElementType.TRANSITION

        if self.config.scene_heading_pattern.match(line.text):
            return ElementType.SCENE_HEADING

        if th.character_min <= x0 <= th.character_max:
            upper = line.text.upper()
            if upper in self.config.section_keywords:
                return ElementType.SECTION_HEADER
            if upper in self.config.continuation_markers:
                return InternalType.PAGE_NUMBER
            return InternalType.CHARACTER

        if th.paren_min <= x0 <= th.paren_max:
            return ElementType.PARENTHETICAL
        if th.dialogue_min <= x0 <= th.dialogue_max:
            return ElementType.DIALOGUE

        if self._is_allcaps_slug(line.text):
            return ElementType.SCENE_HEADING

        if self._is_scene_number(line.text):
            return InternalType.PAGE_NUMBER

        return ElementType.ACTION

    _SCENE_NUMBER_RE = re.compile(r"^\d+[A-Z]?\.?$")

    @classmethod
    def _is_scene_number(cls, text: str) -> bool:
        """Recognise standalone scene numbers like ``1``, ``42``, ``65A``."""
        return bool(cls._SCENE_NUMBER_RE.match(text.strip()))

    _ALLCAPS_SLUG_REJECT = re.compile(r"[(\[\]]|--\s*$|\.{3}")

    def _is_allcaps_slug(self, text: str) -> bool:
        """Detect all-caps text that acts as a non-standard scene heading
        (e.g. ``OVER DARKNESS``, ``ON PIT WALL:``).

        Requires multiple words to avoid false positives from line-wrap
        artifacts (e.g. "TRACK." split from the previous line) and single-word
        emphasis.  Rejects continuations (``--``), ellipsis trails, and
        parenthetical character introductions.
        """
        letters = re.sub(r"[^A-Za-z]", "", text)
        if len(letters) < 4 or letters != letters.upper():
            return False
        words = re.sub(r"[^\w\s]", "", text).split()
        if len(words) < 2:
            return False
        return not self._ALLCAPS_SLUG_REJECT.search(text)

    def _match_clusters_to_roles(self, clusters: list[float]) -> tuple[float, float, float, float]:
        """Match detected clusters to the four screenplay roles (action, dialogue,
        parenthetical, character) by closest distance to the known default positions.

        Handles PDFs with extra indentation levels (e.g. scene numbers in the left
        margin) that would otherwise shift every role assignment.
        """
        defaults = (
            self.config.default_action_x,
            self.config.default_dialogue_x,
            self.config.default_paren_x,
            self.config.default_char_x,
        )
        available = list(clusters)
        assigned: list[float] = []
        for target in defaults:
            best = min(available, key=lambda x: abs(x - target))
            assigned.append(best)
            available.remove(best)
        return assigned[0], assigned[1], assigned[2], assigned[3]

    def _cluster_histogram(self, hist: Counter[int]) -> list[float]:
        clusters: list[tuple[float, int]] = []
        for x in sorted(hist):
            if clusters and x - clusters[-1][0] < 15:
                cx, cc = clusters[-1]
                total = cc + hist[x]
                clusters[-1] = ((cx * cc + x * hist[x]) / total, total)
            else:
                clusters.append((float(x), hist[x]))

        peak = max(c for _, c in clusters)
        valid_clusters = [x for x, c in clusters if c > peak * 0.01]
        return sorted(valid_clusters)

    def _build_thresholds(self, ax: float, dx: float, px: float, cx: float) -> IndentationThresholds:
        return IndentationThresholds(
            action_max=(ax + dx) / 2,
            dialogue_min=(ax + dx) / 2,
            dialogue_max=(dx + px) / 2,
            paren_min=(dx + px) / 2,
            paren_max=(px + cx) / 2,
            character_min=(px + cx) / 2,
            character_max=self.config.x_transition,
        )


# ---------------------------------------------------------------------------
# 5. Screenplay Builder (State Management)
# ---------------------------------------------------------------------------

class ScreenplayBuilder:
    """
    Maintains the state machine of building a document.
    Eliminates complex `nonlocal` closures and giant loops.
    """

    def __init__(self, config: ScreenplayConfig):
        self.config = config
        self.screenplay = Screenplay()

        self.current_scene: Scene | None = None
        self.pending_element: SceneElement | None = None
        self.current_character: str | None = None
        self.in_voice_over: bool = False
        self.seen_page_one_lines: list[str] = []

    def process_line(self, line: RawLine, etype: ElementType | InternalType, canon_map: dict[str, str]) -> None:
        if line.page_num == 1:
            self.seen_page_one_lines.append(line.text)
            return

        if etype == InternalType.CHARACTER:
            self._flush_pending()
            self.in_voice_over = "(V.O.)" in line.text.upper()
            base = self._normalize_character(line.text)
            self.current_character = canon_map.get(base, base)
            return

        if etype == ElementType.DIALOGUE and self.in_voice_over:
            etype = ElementType.VOICE_OVER

        active_character: str | None = None
        if etype in (ElementType.DIALOGUE, ElementType.VOICE_OVER, ElementType.PARENTHETICAL):
            active_character = self.current_character
        else:
            self.current_character = None
            self.in_voice_over = False

        if self._should_merge_with_pending(etype, active_character, line.page_num):
            assert self.pending_element is not None
            self.pending_element.text += " " + line.text
            self.pending_element.bbox = self.pending_element.bbox.merge(line.bbox)
        else:
            self._flush_pending()
            self.pending_element = SceneElement(
                type=etype, text=line.text, page=line.page_num,
                bbox=line.bbox, character=active_character,
            )

    def finalize(self) -> Screenplay:
        self._flush_pending()
        self._parse_title_page()
        return self.screenplay

    def _should_merge_with_pending(self, etype: ElementType, character: str | None, page_num: int) -> bool:
        return (
            self.pending_element is not None
            and self.pending_element.type == etype
            and self.pending_element.character == character
            and self.pending_element.page == page_num
        )

    def _flush_pending(self) -> None:
        if self.pending_element is None:
            return

        self.screenplay.elements.append(self.pending_element)

        if self.pending_element.type == ElementType.SCENE_HEADING:
            self._create_new_scene(self.pending_element)
        elif self.current_scene is not None:
            self.current_scene.elements.append(self.pending_element)

        self.pending_element = None

    _TRAILING_SCENE_NUM = re.compile(r"\s+\d+[A-Z]?$")

    def _create_new_scene(self, element: SceneElement) -> None:
        heading = self._TRAILING_SCENE_NUM.sub("", element.text)
        element.text = heading
        loc_type, location, tod = self._parse_scene_heading(heading)
        self.current_scene = Scene(
            heading=heading, page=element.page,
            location_type=loc_type, location=location, time_of_day=tod,
        )
        self.screenplay.scenes.append(self.current_scene)

    def _normalize_character(self, raw: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", raw).strip()

    def _parse_scene_heading(self, raw: str) -> tuple[LocationType | None, str, str]:
        """
        Parse 'INT. STACK'S SEDAN - DAY' → (INTERIOR, "STACK'S SEDAN", "DAY").

        Handles multi-dash headings by recognizing time-of-day keywords:
          "INT. LUMBER MILL - FIRST FLOOR - DAY"
            → (INTERIOR, "LUMBER MILL - FIRST FLOOR", "DAY")
        """
        upper = raw.upper().strip()
        loc_type: LocationType | None = None

        for prefix in sorted(_LOCATION_TYPE_MAP, key=lambda x: -len(x)):
            if upper.startswith(prefix):
                loc_type = _LOCATION_TYPE_MAP[prefix]
                remainder = raw[len(prefix):].strip()
                break
        else:
            return None, raw, ""

        if " - " not in remainder:
            return loc_type, remainder, ""

        segments = [s.strip() for s in remainder.split(" - ")]

        time_idx: int | None = None
        for i, seg in enumerate(segments):
            if seg.upper() in self.config.time_of_day_keywords:
                time_idx = i
                break

        if time_idx is not None and time_idx > 0:
            location = " - ".join(segments[:time_idx])
            time_of_day = " - ".join(segments[time_idx:])
        else:
            location = " - ".join(segments[:-1])
            time_of_day = segments[-1]

        return loc_type, location, time_of_day

    def _parse_title_page(self) -> None:
        if not self.seen_page_one_lines:
            return

        self.screenplay.title = self.seen_page_one_lines[0]

        author_fragments: list[str] = []
        collecting = False
        for line in self.seen_page_one_lines:
            lower = line.lower().strip()
            if any(lower == kw or lower.startswith(kw + " ") for kw in self.config.credit_keywords):
                if collecting and author_fragments:
                    break
                collecting = True
                continue
            if collecting:
                if not lower:
                    break
                author_fragments.append(line.strip())

        self.screenplay.authors = _parse_authors(" ".join(author_fragments))


# ---------------------------------------------------------------------------
# 6. Helpers
# ---------------------------------------------------------------------------

def _parse_authors(raw: str) -> list[str]:
    """Split a credit string into individual author names."""
    s = raw.strip()
    if not s:
        return []

    s = re.sub(r",\s+and\s+", " | ", s, flags=re.IGNORECASE)
    s = re.sub(r",\s+&\s+", " | ", s)
    s = re.sub(r"\s+&\s+", " | ", s)
    s = re.sub(r"\s+and\s+", " | ", s, flags=re.IGNORECASE)

    parts: list[str] = []
    for chunk in s.split("|"):
        for name in chunk.split(","):
            name = name.strip()
            if name:
                parts.append(name)
    return parts


def _build_character_canon_map(raw_names: list[str]) -> dict[str, str]:
    """
    Build a mapping from every observed character cue to a canonical spelling.

    Groups raw names that become identical when all spaces are removed, then
    picks the most frequent variant as canonical.  This corrects OCR artifacts
    like "SA MMIE" → "SAMMIE" while preserving intentional multi-word names.
    """
    freq: Counter[str] = Counter(raw_names)

    groups: dict[str, list[str]] = {}
    for name in freq:
        key = name.replace(" ", "")
        groups.setdefault(key, []).append(name)

    canon: dict[str, str] = {}
    for key, variants in groups.items():
        best = max(variants, key=lambda v: freq[v])
        if len(variants) == 1 and " " in best and freq[best] <= 2:
            parts = best.split(" ")
            if (
                len(parts) == 2
                and all(p.isalpha() for p in parts)
                and min(len(parts[0]), len(parts[1])) <= 2
            ):
                best = key
        for v in variants:
            canon[v] = best
    return canon


# ---------------------------------------------------------------------------
# 6. Parse errors (for API to surface to user)
# ---------------------------------------------------------------------------


class ScreenplayParseError(Exception):
    """Raised when a PDF could not be parsed as a screenplay (e.g. no structure extracted)."""


# ---------------------------------------------------------------------------
# 7. Main Orchestrator
# ---------------------------------------------------------------------------

def parse_screenplay(pdf_path: str | Path, max_pages: int | None = None) -> Screenplay:
    """
    The orchestrator function. Wires up IO, Layout Analysis, and Document Assembly.
    If max_pages is set, only the first max_pages pages of the PDF are parsed.
    """
    path = Path(pdf_path)
    ctx = _tracer.start_as_current_span("screenplay.parse") if _tracer else _null_span()
    with ctx:
        span = trace.get_current_span() if (trace is not None and _tracer) else None
        if span is not None and span.is_recording():
            span.set_attribute("screenplay.pdf_path", str(path))

        config = ScreenplayConfig()
        analyzer = LayoutAnalyzer(config)
        builder = ScreenplayBuilder(config)

        raw_lines = list(extract_pdf_lines(path, max_pages=max_pages))

        if not raw_lines:
            raise ScreenplayParseError(
                "PDF has no extractable text. It may be empty, scanned (image-only), or corrupted."
            )

        if span is not None and span.is_recording():
            span.set_attribute("screenplay.raw_line_count", len(raw_lines))

        build_ctx = _tracer.start_as_current_span("screenplay.analyze_and_build") if _tracer else _null_span()
        with build_ctx:
            thresholds = analyzer.detect_thresholds(raw_lines)

            classified_data: list[tuple[RawLine, ElementType | InternalType]] = []
            raw_chars: list[str] = []

            for line in raw_lines:
                clean_text = re.sub(r"  +", " ", line.text)
                clean_line = RawLine(line.page_num, clean_text, line.bbox)
                etype = analyzer.classify_line(clean_line, thresholds)

                if etype == InternalType.PAGE_NUMBER:
                    continue

                if etype == InternalType.CHARACTER:
                    raw_chars.append(builder._normalize_character(clean_text))

                classified_data.append((clean_line, etype))

            canon_map = _build_character_canon_map(raw_chars)

            for line, etype in classified_data:
                builder.process_line(line, etype, canon_map)

            screenplay = builder.finalize()

        if not screenplay.scenes and not screenplay.elements:
            raise ScreenplayParseError(
                "No screenplay structure could be extracted. The PDF may be a scan, use an older or "
                "unsupported script format, or not be a standard screenplay."
            )
        if span is not None and span.is_recording():
            span.set_attribute("screenplay.scene_count", len(screenplay.scenes))
            span.set_attribute("screenplay.element_count", len(screenplay.elements))
            span.set_attribute("screenplay.title", screenplay.title or "")
        return screenplay
