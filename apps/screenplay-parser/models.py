"""
SQLAlchemy ORM models for screenplay and RAG chunk storage (pgvector).

Industry-standard SQLAlchemy 2.x with Mapped/mapped_column; vector column via pgvector.
Screenplay id is UUID v7 (time-ordered, globally unique).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

import uuid7
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

if TYPE_CHECKING:
    from sqlalchemy.orm import relationship as Relationship

# Embedding dimension; match your embedding model (e.g. OpenAI text-embedding-3-small = 1536)
EMBEDDING_DIM = 1536


class LLMReason(str, enum.Enum):
    """Reason for the LLM request; used for cost tracking."""

    QA_ANSWER = "qa_answer"           # generate_content for user Q&A
    EMBED_INGEST = "embed_ingest"     # embed_content during screenplay ingest
    EMBED_QUERY = "embed_query"       # embed_content for semantic search (e.g. during QA)


class ScreenplayModel(Base):
    """Stored screenplay document (one per uploaded PDF)."""

    __tablename__ = "screenplays"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7.create
    )
    title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    authors: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)  # list[str]
    source_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # full screenplay JSON for viewer
    is_deleted: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scenes: Mapped[list["SceneModel"]] = relationship(
        "SceneModel", back_populates="screenplay", order_by="SceneModel.scene_index"
    )
    chunks: Mapped[list["RAGChunkModel"]] = relationship(
        "RAGChunkModel", back_populates="screenplay", cascade="all, delete-orphan"
    )
    llm_costs: Mapped[list["ScreenplayLLMCostModel"]] = relationship(
        "ScreenplayLLMCostModel", back_populates="screenplay"
    )


class ScreenplayLLMCostModel(Base):
    """One LLM (Gemini) request: tokens and optional cost, keyed by reason."""

    __tablename__ = "screenplay_llm_costs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    screenplay_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("screenplays.id", ondelete="CASCADE"), nullable=True, index=True
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)  # LLMReason.value
    input_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    cost: Mapped[float | None] = mapped_column(nullable=True)  # optional $ cost
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    screenplay: Mapped["ScreenplayModel | None"] = relationship(
        "ScreenplayModel", back_populates="llm_costs"
    )


class SceneModel(Base):
    """One scene within a screenplay (INT/EXT heading + location + time)."""

    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    screenplay_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("screenplays.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_index: Mapped[int] = mapped_column(nullable=False)  # 0-based order
    heading: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int] = mapped_column(nullable=False)
    location_type: Mapped[str | None] = mapped_column(Text, nullable=True)  # INT, EXT, I/E
    location: Mapped[str] = mapped_column(Text, default="", nullable=False)
    time_of_day: Mapped[str] = mapped_column(Text, default="", nullable=False)

    screenplay: Mapped["ScreenplayModel"] = relationship("ScreenplayModel", back_populates="scenes")
    chunks: Mapped[list["RAGChunkModel"]] = relationship(
        "RAGChunkModel", back_populates="scene", cascade="all, delete-orphan"
    )


class RAGChunkModel(Base):
    """One RAG chunk: scene or dialogue block text + embedding for similarity search."""

    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    screenplay_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("screenplays.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenes.id", ondelete="CASCADE"), nullable=True, index=True
    )
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False)  # "scene" | "dialogue"
    chunk_index: Mapped[int] = mapped_column(nullable=False)  # order within screenplay
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )  # pgvector
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )  # scene_heading, page, characters, etc.

    screenplay: Mapped["ScreenplayModel"] = relationship(
        "ScreenplayModel", back_populates="chunks"
    )
    scene: Mapped["SceneModel | None"] = relationship(
        "SceneModel", back_populates="chunks"
    )


# HNSW index for fast approximate nearest-neighbor search
Index(
    "ix_rag_chunks_embedding_hnsw",
    RAGChunkModel.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
