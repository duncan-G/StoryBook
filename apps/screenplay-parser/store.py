"""
Persist screenplay and RAG chunks to Postgres (SQLAlchemy + pgvector).

Embedding is pluggable: pass an embed_fn(texts: list[str]) -> list[list[float]]
or leave None to store chunks without vectors (e.g. embed later or use keyword search).
"""

from __future__ import annotations

import uuid
from typing import Callable

from opentelemetry import trace
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from chunking import RAGChunk, build_rag_chunks
from config import EMBEDDING_COST_PER_MILLION_TOKENS
from models import (
    LLMReason,
    RAGChunkModel,
    SceneModel,
    ScreenplayLLMCostModel,
    ScreenplayModel,
)

# Type for embedding function: list of texts -> list of vectors (each list[float])
EmbedFn = Callable[[list[str]], list[list[float]]]

tracer = trace.get_tracer(__name__, "0.1.0")


async def record_llm_cost(
    session: AsyncSession,
    reason: LLMReason,
    input_tokens: int,
    output_tokens: int,
    *,
    screenplay_id: uuid.UUID | str | None = None,
    cost: float | None = None,
) -> None:
    """Record one LLM (Gemini) request for cost tracking."""
    sid = None
    if screenplay_id is not None:
        sid = screenplay_id if isinstance(screenplay_id, uuid.UUID) else uuid.UUID(screenplay_id)
    session.add(
        ScreenplayLLMCostModel(
            screenplay_id=sid,
            reason=reason.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
    )


async def store_screenplay(
    session: AsyncSession,
    screenplay,  # domain Screenplay from screenplay_parser
    *,
    source_filename: str | None = None,
    content_json: dict | None = None,
    embed_fn: EmbedFn | None = None,
    scene_level: bool = True,
    dialogue_level: bool = False,
) -> uuid.UUID:
    """
    Persist a parsed screenplay and its RAG chunks to the database.

    content_json: optional full screenplay JSON (for GET by id / viewer).
    Returns the screenplay id (UUID). If embed_fn is provided, embeddings are computed
    and stored; otherwise embedding column is left null.
    """
    from screenplay_parser import Screenplay as DomainScreenplay

    with tracer.start_as_current_span("store_screenplay") as span:
        span.set_attribute("store.source_filename", source_filename or "")
        span.set_attribute("store.embeddings_enabled", embed_fn is not None)
        span.set_attribute("store.scene_level", scene_level)
        span.set_attribute("store.dialogue_level", dialogue_level)

        sp = screenplay
        db_screenplay = ScreenplayModel(
            title=sp.title or "",
            authors=list(sp.authors),
            source_filename=source_filename,
            content=content_json,
        )
        session.add(db_screenplay)
        await session.flush()  # get db_screenplay.id

        # Insert scenes
        scene_id_by_index: dict[int, int] = {}
        for idx, scene in enumerate(sp.scenes):
            db_scene = SceneModel(
                screenplay_id=db_screenplay.id,
                scene_index=idx,
                heading=scene.heading,
                page=scene.page,
                location_type=scene.location_type.value if scene.location_type else None,
                location=scene.location or "",
                time_of_day=scene.time_of_day or "",
            )
            session.add(db_scene)
            await session.flush()
            scene_id_by_index[idx] = db_scene.id

        # Build chunks and optionally embed
        chunks = build_rag_chunks(
            sp, scene_level=scene_level, dialogue_level=dialogue_level
        )
        span.set_attribute("store.chunk_count", len(chunks))
        span.set_attribute("store.scene_count", len(sp.scenes))

        texts = [c.text for c in chunks]
        embeddings: list[list[float] | None] = [None] * len(texts)
        if embed_fn and texts:
            result = embed_fn(texts, session=session, screenplay_id=db_screenplay.id)
            if isinstance(result, tuple):
                embeddings, usage = result[0], result[1]
                if usage:
                    inp = int(usage.get("input_tokens", 0))
                    out = int(usage.get("output_tokens", 0))
                    cost = (inp + out) / 1_000_000 * EMBEDDING_COST_PER_MILLION_TOKENS
                    await record_llm_cost(
                        session,
                        LLMReason.EMBED_INGEST,
                        inp,
                        out,
                        screenplay_id=db_screenplay.id,
                        cost=cost,
                    )
            else:
                embeddings = result

        for c, emb in zip(chunks, embeddings):
            scene_id = scene_id_by_index.get(c.scene_index)
            meta = {
                "scene_index": c.scene_index,
                "scene_heading": c.scene_heading,
                "page": c.page,
                "location": c.location,
                "time_of_day": c.time_of_day,
                "characters": c.characters,
            }
            if c.character is not None:
                meta["character"] = c.character
                meta["is_voice_over"] = c.is_voice_over
            session.add(
                RAGChunkModel(
                    screenplay_id=db_screenplay.id,
                    scene_id=scene_id,
                    chunk_type=c.chunk_type,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    embedding=emb,
                    metadata_=meta,
                )
            )
        await session.flush()
        span.set_attribute("store.screenplay_id", str(db_screenplay.id))
        return db_screenplay.id


async def search_chunks(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    screenplay_id: uuid.UUID | str | None = None,
    limit: int = 10,
) -> list[tuple[str, dict, int | None]]:
    """
    Cosine similarity search over RAG chunks. Returns list of (text, metadata, scene_index).
    scene_index comes from the joined scenes table (or metadata fallback).
    """
    distance = RAGChunkModel.embedding.cosine_distance(query_embedding)
    q = (
        select(RAGChunkModel.text, RAGChunkModel.metadata_, SceneModel.scene_index)
        .outerjoin(SceneModel, RAGChunkModel.scene_id == SceneModel.id)
        .where(RAGChunkModel.embedding.isnot(None))
        .order_by(distance)
        .limit(limit)
    )
    if screenplay_id is not None:
        sid = screenplay_id if isinstance(screenplay_id, uuid.UUID) else uuid.UUID(screenplay_id)
        q = q.where(RAGChunkModel.screenplay_id == sid)
    result = await session.execute(q)
    rows = []
    for row in result.all():
        text, meta, scene_index = row[0], row[1] or {}, row[2]
        if scene_index is None:
            scene_index = meta.get("scene_index")
        rows.append((text, meta, scene_index))
    return rows


# ---------------------------------------------------------------------------
# Query helpers for Gemini QA (function calling)
# ---------------------------------------------------------------------------


async def get_screenplay_content(
    session: AsyncSession, screenplay_id: uuid.UUID | str
) -> dict | None:
    """
    Return full screenplay JSON for viewer (content column). None if not found or no content.
    """
    try:
        sid = screenplay_id if isinstance(screenplay_id, uuid.UUID) else uuid.UUID(screenplay_id)
    except (ValueError, TypeError):
        return None
    q = (
        select(ScreenplayModel.content)
        .where(ScreenplayModel.id == sid)
        .where(ScreenplayModel.is_deleted == False)
    )
    result = await session.execute(q)
    row = result.one_or_none()
    return row[0] if row and row[0] is not None else None


async def soft_delete_screenplay(
    session: AsyncSession, screenplay_id: uuid.UUID | str
) -> bool:
    """
    Soft-delete a screenplay by setting is_deleted=True.
    Returns True if a row was updated, False if not found.
    """
    try:
        sid = screenplay_id if isinstance(screenplay_id, uuid.UUID) else uuid.UUID(screenplay_id)
    except (ValueError, TypeError):
        return False
    result = await session.execute(
        update(ScreenplayModel)
        .where(ScreenplayModel.id == sid)
        .where(ScreenplayModel.is_deleted == False)
        .values(is_deleted=True)
    )
    return result.rowcount > 0


async def list_screenplays(session: AsyncSession) -> list[dict]:
    """
    List all stored screenplays (excluding soft-deleted). Returns list of {id, title, authors, source_filename}.
    """
    q = (
        select(
            ScreenplayModel.id,
            ScreenplayModel.title,
            ScreenplayModel.authors,
            ScreenplayModel.source_filename,
        )
        .where(ScreenplayModel.is_deleted == False)
        .order_by(ScreenplayModel.id)
    )
    result = await session.execute(q)
    return [
        {
            "id": str(row[0]),
            "title": row[1] or "",
            "authors": row[2] or [],
            "source_filename": row[3],
        }
        for row in result.all()
    ]


async def get_llm_costs(session: AsyncSession) -> dict:
    """
    Aggregate LLM cost data: totals by reason, per-screenplay breakdown, and recent entries.
    """
    from sqlalchemy import func as sqlfunc

    # --- Per-reason totals ---
    reason_q = (
        select(
            ScreenplayLLMCostModel.reason,
            sqlfunc.count().label("request_count"),
            sqlfunc.sum(ScreenplayLLMCostModel.input_tokens).label("input_tokens"),
            sqlfunc.sum(ScreenplayLLMCostModel.output_tokens).label("output_tokens"),
            sqlfunc.coalesce(sqlfunc.sum(ScreenplayLLMCostModel.cost), 0).label("cost"),
        )
        .group_by(ScreenplayLLMCostModel.reason)
    )
    reason_result = await session.execute(reason_q)
    by_reason = [
        {
            "reason": row.reason,
            "request_count": int(row.request_count),
            "input_tokens": int(row.input_tokens),
            "output_tokens": int(row.output_tokens),
            "cost": float(row.cost),
        }
        for row in reason_result.all()
    ]

    # --- Grand totals ---
    totals_q = select(
        sqlfunc.count().label("request_count"),
        sqlfunc.coalesce(sqlfunc.sum(ScreenplayLLMCostModel.input_tokens), 0).label("input_tokens"),
        sqlfunc.coalesce(sqlfunc.sum(ScreenplayLLMCostModel.output_tokens), 0).label("output_tokens"),
        sqlfunc.coalesce(sqlfunc.sum(ScreenplayLLMCostModel.cost), 0).label("cost"),
    )
    totals_result = await session.execute(totals_q)
    totals_row = totals_result.one()

    # --- Per-screenplay breakdown ---
    sp_q = (
        select(
            ScreenplayLLMCostModel.screenplay_id,
            ScreenplayModel.title,
            sqlfunc.count().label("request_count"),
            sqlfunc.sum(ScreenplayLLMCostModel.input_tokens).label("input_tokens"),
            sqlfunc.sum(ScreenplayLLMCostModel.output_tokens).label("output_tokens"),
            sqlfunc.coalesce(sqlfunc.sum(ScreenplayLLMCostModel.cost), 0).label("cost"),
        )
        .outerjoin(ScreenplayModel, ScreenplayLLMCostModel.screenplay_id == ScreenplayModel.id)
        .where(ScreenplayLLMCostModel.screenplay_id.isnot(None))
        .group_by(ScreenplayLLMCostModel.screenplay_id, ScreenplayModel.title)
        .order_by(sqlfunc.sum(ScreenplayLLMCostModel.cost).desc())
    )
    sp_result = await session.execute(sp_q)
    by_screenplay = [
        {
            "screenplay_id": str(row.screenplay_id),
            "title": row.title or "Untitled",
            "request_count": int(row.request_count),
            "input_tokens": int(row.input_tokens),
            "output_tokens": int(row.output_tokens),
            "cost": float(row.cost),
        }
        for row in sp_result.all()
    ]

    # --- Recent entries (last 50) ---
    recent_q = (
        select(
            ScreenplayLLMCostModel.id,
            ScreenplayLLMCostModel.screenplay_id,
            ScreenplayModel.title,
            ScreenplayLLMCostModel.reason,
            ScreenplayLLMCostModel.input_tokens,
            ScreenplayLLMCostModel.output_tokens,
            ScreenplayLLMCostModel.cost,
            ScreenplayLLMCostModel.created_at,
        )
        .outerjoin(ScreenplayModel, ScreenplayLLMCostModel.screenplay_id == ScreenplayModel.id)
        .order_by(ScreenplayLLMCostModel.created_at.desc())
        .limit(50)
    )
    recent_result = await session.execute(recent_q)
    recent = [
        {
            "id": row.id,
            "screenplay_id": str(row.screenplay_id) if row.screenplay_id else None,
            "screenplay_title": row.title or "Untitled" if row.screenplay_id else None,
            "reason": row.reason,
            "input_tokens": int(row.input_tokens),
            "output_tokens": int(row.output_tokens),
            "cost": float(row.cost) if row.cost is not None else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in recent_result.all()
    ]

    return {
        "totals": {
            "request_count": int(totals_row.request_count),
            "input_tokens": int(totals_row.input_tokens),
            "output_tokens": int(totals_row.output_tokens),
            "cost": float(totals_row.cost),
        },
        "by_reason": by_reason,
        "by_screenplay": by_screenplay,
        "recent": recent,
    }


async def get_screenplay_scenes(
    session: AsyncSession, screenplay_id: uuid.UUID | str
) -> list[dict]:
    """
    Get all scenes for a screenplay. Returns list of {scene_index, heading, page, location, time_of_day}.
    """
    sid = screenplay_id if isinstance(screenplay_id, uuid.UUID) else uuid.UUID(screenplay_id)
    q = (
        select(
            SceneModel.scene_index,
            SceneModel.heading,
            SceneModel.page,
            SceneModel.location,
            SceneModel.time_of_day,
        )
        .where(SceneModel.screenplay_id == sid)
        .order_by(SceneModel.scene_index)
    )
    result = await session.execute(q)
    return [
        {
            "scene_index": row[0],
            "heading": row[1],
            "page": row[2],
            "location": row[3] or "",
            "time_of_day": row[4] or "",
        }
        for row in result.all()
    ]
