"""
Screenplay → RAG chunks: scene-level (and optional dialogue-level) chunking.

Produces (text, metadata) pairs for embedding and storage in pgvector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from screenplay_parser import Scene, Screenplay


@dataclass
class RAGChunk:
    """One chunk of text plus metadata for RAG storage/retrieval."""

    chunk_type: str  # "scene" | "dialogue"
    chunk_index: int
    text: str
    scene_index: int
    scene_heading: str
    page: int
    location: str
    time_of_day: str
    characters: list[str]
    # Optional for dialogue chunks
    character: str | None = None
    is_voice_over: bool = False


def _scene_to_text(scene: "Scene") -> str:
    """Render a single scene as continuous text: heading, action, dialogue."""
    parts = [scene.heading]
    if scene.location or scene.time_of_day:
        loc_tod = [s for s in (scene.location, scene.time_of_day) if s]
        if loc_tod:
            parts.append(" ".join(loc_tod))
    parts.extend(scene.action_lines)
    for block in scene.dialogue_blocks:
        prefix = f"{block.character} (V.O.): " if block.is_voice_over else f"{block.character}: "
        parts.append(prefix + block.speech)
        if block.parentheticals:
            parts.append(" (" + "; ".join(block.parentheticals) + ")")
    return "\n\n".join(p for p in parts if p)


def build_scene_chunks(screenplay: "Screenplay") -> list[RAGChunk]:
    """
    One chunk per scene (semantic unit). Best default for RAG: preserves
    scene boundaries and keeps action + dialogue together.
    """
    chunks: list[RAGChunk] = []
    for idx, scene in enumerate(screenplay.scenes):
        text = _scene_to_text(scene)
        if not text.strip():
            continue
        chunks.append(
            RAGChunk(
                chunk_type="scene",
                chunk_index=len(chunks),
                text=text,
                scene_index=idx,
                scene_heading=scene.heading,
                page=scene.page,
                location=scene.location or "",
                time_of_day=scene.time_of_day or "",
                characters=sorted(scene.characters_present),
            )
        )
    return chunks


def build_dialogue_chunks(screenplay: "Screenplay") -> list[RAGChunk]:
    """
    One chunk per dialogue block (finer granularity). Use when you need
    character-specific or line-level retrieval.
    """
    chunks: list[RAGChunk] = []
    for scene_idx, scene in enumerate(screenplay.scenes):
        for block in scene.dialogue_blocks:
            speech = block.speech.strip()
            if not speech:
                continue
            prefix = f"{block.character} (V.O.): " if block.is_voice_over else f"{block.character}: "
            text = f"{scene.heading}\n\n{prefix}{speech}"
            if block.parentheticals:
                text += "\n(" + "; ".join(block.parentheticals) + ")"
            chunks.append(
                RAGChunk(
                    chunk_type="dialogue",
                    chunk_index=len(chunks),
                    text=text,
                    scene_index=scene_idx,
                    scene_heading=scene.heading,
                    page=scene.page,
                    location=scene.location or "",
                    time_of_day=scene.time_of_day or "",
                    characters=sorted(scene.characters_present),
                    character=block.character,
                    is_voice_over=block.is_voice_over,
                )
            )
    return chunks


def build_rag_chunks(
    screenplay: "Screenplay",
    *,
    scene_level: bool = True,
    dialogue_level: bool = False,
) -> list[RAGChunk]:
    """
    Build RAG chunks from a parsed screenplay.

    - scene_level=True: one chunk per scene (default, recommended).
    - dialogue_level=True: add one chunk per dialogue block (in addition or alone).
    """
    if dialogue_level and not scene_level:
        return build_dialogue_chunks(screenplay)
    if scene_level and not dialogue_level:
        return build_scene_chunks(screenplay)
    # Both: scene chunks first, then dialogue chunks with adjusted chunk_index
    scene_chunks = build_scene_chunks(screenplay)
    dialogue_chunks = build_dialogue_chunks(screenplay)
    for i, c in enumerate(dialogue_chunks):
        c.chunk_index = len(scene_chunks) + i
    return scene_chunks + dialogue_chunks
