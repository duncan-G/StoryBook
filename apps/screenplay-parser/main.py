"""
FastAPI service for screenplay PDF parsing.

Upload a PDF file and receive a structured JSON representation
of the screenplay (scenes, dialogue, action, characters, etc.).
Optionally ingest into Postgres with pgvector for RAG (see /ingest).
"""

from __future__ import annotations

import logging
import os
import uuid

# Load .env before any other imports that read environment variables
from dotenv import load_dotenv
load_dotenv()

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from opentelemetry import trace
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_factory, engine, init_db
from screenplay_parser import (
    DialogueBlock as DomainDialogueBlock,
    Scene as DomainScene,
    Screenplay as DomainScreenplay,
    ScreenplayParseError,
    parse_screenplay,
)
from gemini_qa import answer_question
from store import get_llm_costs, get_screenplay_content, list_screenplays, soft_delete_screenplay, store_screenplay
from telemetry import instrument_app

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__, "0.1.0")

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class BoundingBoxResponse(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class SceneElementResponse(BaseModel):
    type: str
    text: str
    page: int
    bbox: BoundingBoxResponse
    character: str | None = None


class DialogueBlockResponse(BaseModel):
    character: str
    is_voice_over: bool = False
    speech: str = ""
    parentheticals: list[str] = Field(default_factory=list)
    lines: list[SceneElementResponse] = Field(default_factory=list)


class SceneResponse(BaseModel):
    heading: str
    page: int
    location_type: str | None = None
    location: str = ""
    time_of_day: str = ""
    action_lines: list[str] = Field(default_factory=list)
    characters_present: list[str] = Field(default_factory=list)
    dialogue_blocks: list[DialogueBlockResponse] = Field(default_factory=list)
    elements: list[SceneElementResponse] = Field(default_factory=list)


class ScreenplayResponse(BaseModel):
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    all_characters: list[str] = Field(default_factory=list)
    scenes: list[SceneResponse] = Field(default_factory=list)
    elements: list[SceneElementResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_element(el) -> SceneElementResponse:
    return SceneElementResponse(
        type=el.type.name,
        text=el.text,
        page=el.page,
        bbox=BoundingBoxResponse(x0=el.bbox.x0, y0=el.bbox.y0, x1=el.bbox.x1, y1=el.bbox.y1),
        character=el.character,
    )


def _serialize_dialogue_block(block: DomainDialogueBlock) -> DialogueBlockResponse:
    return DialogueBlockResponse(
        character=block.character,
        is_voice_over=block.is_voice_over,
        speech=block.speech,
        parentheticals=block.parentheticals,
        lines=[_serialize_element(el) for el in block.lines],
    )


def _serialize_scene(scene: DomainScene) -> SceneResponse:
    return SceneResponse(
        heading=scene.heading,
        page=scene.page,
        location_type=scene.location_type.value if scene.location_type else None,
        location=scene.location,
        time_of_day=scene.time_of_day,
        action_lines=scene.action_lines,
        characters_present=sorted(scene.characters_present),
        dialogue_blocks=[_serialize_dialogue_block(b) for b in scene.dialogue_blocks],
        elements=[_serialize_element(el) for el in scene.elements],
    )


def _serialize_screenplay(sp: DomainScreenplay) -> ScreenplayResponse:
    return ScreenplayResponse(
        title=sp.title,
        authors=sp.authors,
        all_characters=sorted(sp.all_characters),
        scenes=[_serialize_scene(s) for s in sp.scenes],
        elements=[_serialize_element(el) for el in sp.elements],
    )


# ---------------------------------------------------------------------------
# DB session dependency & lifespan
# ---------------------------------------------------------------------------

async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Screenplay parser started")
    yield
    logger.info("Screenplay parser shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Screenplay PDF Parser",
    description="Upload a screenplay PDF and receive a structured JSON representation. Ingest into Postgres+pgvector for RAG.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow client origin(s). Defaults include localhost and 127.0.0.1 (either can be used in dev).
_cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrument_app(app, engine=engine)


def _get_embed_fn():
    """Optional embedding function; set GOOGLE_API_KEY or GEMINI_API_KEY to use Gemini embeddings."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types
        from google.genai.local_tokenizer import LocalTokenizer

        client = genai.Client(api_key=api_key)
        # Use 1536 to match EMBEDDING_DIM in models.py (pgvector column size)
        output_dim = 1536

        # Gemini allows at most 100 requests per batch
        _BATCH_SIZE = 100

        # Local tokenizer fallback when API doesn't return token_count (e.g. non-Vertex)
        _local_tok: LocalTokenizer | None = None

        def _token_count_from_batch(
            result,
            batch_texts: list[str],
        ) -> int:
            """Input token count from embeddings[0].statistics.token_count (or sum across embeddings), or local tokenizer if null."""
            if hasattr(result, "embeddings") and result.embeddings:
                # Prefer first embedding's token_count (API may return aggregate there)
                first = result.embeddings[0]
                st = getattr(first, "statistics", None)
                if st is not None:
                    tc = getattr(st, "token_count", None)
                    if tc is not None:
                        return int(tc)
                # Else sum per-embedding token_count when available
                total = 0
                for e in result.embeddings:
                    st = getattr(e, "statistics", None)
                    if st is not None:
                        tc = getattr(st, "token_count", None)
                        if tc is not None:
                            total += int(tc)
                if total > 0:
                    return total
            nonlocal _local_tok
            if _local_tok is None:
                _local_tok = LocalTokenizer(model_name="gemini-2.5-flash")
            contents = [types.UserContent(t) for t in batch_texts]
            count_result = _local_tok.count_tokens(contents)
            return int(getattr(count_result, "total_tokens", 0) or 0)

        def embed(
            texts: list[str],
            *,
            session=None,
            screenplay_id=None,
        ):
            if not texts:
                return ([], {"input_tokens": 0, "output_tokens": 0}) if session is not None else []
            out: list[list[float]] = []
            total_in = 0
            for i in range(0, len(texts), _BATCH_SIZE):
                batch = texts[i : i + _BATCH_SIZE]
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=output_dim),
                )
                if hasattr(result, "embeddings") and result.embeddings:
                    out.extend(e.values for e in result.embeddings)
                if session is not None:
                    total_in += _token_count_from_batch(result, batch)
            if session is not None:
                return (out, {"input_tokens": total_in, "output_tokens": 0})
            return out
        return embed
    except ImportError as e:
        logger.error("Error importing Google GenAI: %s", e)
        return None


@app.post("/parse", response_model=ScreenplayResponse)
async def parse_uploaded_pdf(file: UploadFile):
    """
    Accept a PDF upload, parse it as a screenplay, and return structured JSON.
    """
    filename = file.filename or "(none)"
    logger.info("Parse request: filename=%s", filename)
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tracer.start_as_current_span("parse_screenplay_request") as span:
        span.set_attribute("filename", filename)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            contents = await file.read()
            span.set_attribute("file_size_bytes", len(contents))
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(contents)
            tmp.flush()

            try:
                screenplay = parse_screenplay(Path(tmp.name))
            except ScreenplayParseError as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise HTTPException(status_code=422, detail=str(e))

        span.set_attribute("screenplay.scene_count", len(screenplay.scenes))
        span.set_attribute("screenplay.element_count", len(screenplay.elements))
        span.set_attribute("screenplay.title", screenplay.title or "")
        return _serialize_screenplay(screenplay)


# ---------------------------------------------------------------------------
# Ingest to Postgres + pgvector for RAG
# ---------------------------------------------------------------------------


class ScreenplayListItem(BaseModel):
    id: str  # UUID v7
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    source_filename: str | None = None


class IngestResponse(BaseModel):
    screenplay_id: str  # UUID v7
    message: str = "Screenplay and RAG chunks stored. Set GOOGLE_API_KEY or GEMINI_API_KEY to store embeddings."
    screenplay: ScreenplayResponse | None = None  # full content for viewer


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
):
    """
    Parse the PDF as a screenplay and store it in Postgres with pgvector.
    Chunks are scene-level by default. If GOOGLE_API_KEY or GEMINI_API_KEY is set, embeddings
    are computed with Gemini and stored for similarity search.
    """
    filename = file.filename or None
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tracer.start_as_current_span("ingest_pdf") as span:
        span.set_attribute("filename", filename or "(none)")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            contents = await file.read()
            span.set_attribute("file_size_bytes", len(contents))
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(contents)
            tmp.flush()
            with tracer.start_as_current_span("ingest.parse_screenplay") as parse_span:
                try:
                    screenplay = parse_screenplay(Path(tmp.name))
                except ScreenplayParseError as e:
                    parse_span.record_exception(e)
                    parse_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise HTTPException(status_code=422, detail=str(e))
                parse_span.set_attribute("screenplay.scene_count", len(screenplay.scenes))
                parse_span.set_attribute("screenplay.title", screenplay.title or "")

        response = _serialize_screenplay(screenplay)
        embed_fn = _get_embed_fn()
        with tracer.start_as_current_span("ingest.store_screenplay") as store_span:
            store_span.set_attribute("embeddings_enabled", embed_fn is not None)
            sid = await store_screenplay(
                session,
                screenplay,
                source_filename=filename,
                content_json=response.model_dump(),
                embed_fn=embed_fn,
                scene_level=True,
                dialogue_level=False,
            )
            store_span.set_attribute("screenplay_id", str(sid))
            store_span.set_attribute("scene_count", len(screenplay.scenes))

        span.set_attribute("screenplay_id", str(sid))
        span.set_attribute("embeddings_stored", embed_fn is not None)
        return IngestResponse(
            screenplay_id=str(sid),
            message="Screenplay and RAG chunks stored."
            + (" Embeddings stored." if embed_fn else " Set GOOGLE_API_KEY or GEMINI_API_KEY to store embeddings."),
            screenplay=response,
        )


# ---------------------------------------------------------------------------
# Q&A: ask questions about screenplays (Gemini + function calling)
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the screenplay(s).")
    screenplay_id: str | None = Field(None, description="Optional. Restrict context to this screenplay ID (UUID).")


class SceneReferenceResponse(BaseModel):
    scene_index: int = Field(..., description="0-based scene index in the screenplay.")
    quote: str = Field("", description="Short exact quote from the screenplay text.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Gemini's answer with inline [[ref:N \"quote\"]] citation markers.")
    references: list[SceneReferenceResponse] = Field(default_factory=list, description="Parsed scene references extracted from the answer.")


# ---------------------------------------------------------------------------
# List and get screenplays (for client landing + viewer)
# ---------------------------------------------------------------------------


@app.get("/screenplays", response_model=list[ScreenplayListItem])
async def list_screenplays_api(session: AsyncSession = Depends(get_session)):
    """List all uploaded (stored) screenplays. Excludes soft-deleted (is_deleted=true)."""
    rows = await list_screenplays(session)
    return [ScreenplayListItem(**r) for r in rows]


@app.get("/screenplays/{screenplay_id}", response_model=ScreenplayResponse)
async def get_screenplay_api(
    screenplay_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Return full screenplay JSON for the viewer. 404 if not found or no content stored."""
    try:
        uuid.UUID(screenplay_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid screenplay ID: must be a valid UUID (e.g. from /screenplays list).",
        )
    content = await get_screenplay_content(session, screenplay_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Screenplay not found or content not available.")
    return ScreenplayResponse.model_validate(content)


@app.delete("/screenplays/{screenplay_id}")
async def delete_screenplay_api(
    screenplay_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Soft-delete a screenplay (sets is_deleted=True). Returns 404 if not found."""
    try:
        uuid.UUID(screenplay_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid screenplay ID: must be a valid UUID.",
        )
    deleted = await soft_delete_screenplay(session, screenplay_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Screenplay not found or already deleted.")
    return {"ok": True, "message": "Screenplay deleted."}


@app.get("/costs")
async def get_costs_api(session: AsyncSession = Depends(get_session)):
    """Return aggregated LLM cost data: totals, by reason, by screenplay, and recent entries."""
    return await get_llm_costs(session)


@app.post("/ask", response_model=AskResponse)
async def ask_about_screenplay(
    body: AskRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Answer a question about stored screenplays using Gemini with function calling.
    Gemini can list screenplays, get scenes, and run semantic search over content.
    Set GOOGLE_API_KEY or GEMINI_API_KEY. For semantic search, screenplays must
    have been ingested with embeddings (same env var).

    The answer may contain inline citation markers like [[ref:3 "exact quote"]].
    Parsed references are also returned in the `references` field.
    """
    with tracer.start_as_current_span("ask_question") as span:
        span.set_attribute("question", body.question)
        if body.screenplay_id:
            span.set_attribute("screenplay_id", body.screenplay_id)

        embed_fn = _get_embed_fn()
        span.set_attribute("embeddings_available", embed_fn is not None)

        result = await answer_question(
            session,
            body.question,
            screenplay_id=body.screenplay_id,
            embed_fn=embed_fn,
        )

        span.set_attribute("answer_length", len(result.answer))
        span.set_attribute("reference_count", len(result.references))
        span.set_attribute("tool_rounds", result.tool_rounds)
        span.set_attribute("total_input_tokens", result.total_input_tokens)
        span.set_attribute("total_output_tokens", result.total_output_tokens)
        span.set_attribute("tool_calls", [
            f"{tc['name']}({tc['args']})" for tc in result.tool_call_log
        ])

        return AskResponse(
            answer=result.answer,
            references=[
                SceneReferenceResponse(scene_index=r.scene_index, quote=r.quote)
                for r in result.references
            ],
        )
