"""
Screenplay Q&A using Gemini with function calling.

Gemini can call tools to list screenplays, get scenes, and run semantic search
over RAG chunks. The agent loop runs until the model returns a final text answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from google import genai
from google.genai import types
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    EMBEDDING_COST_PER_MILLION_TOKENS,
    QA_INPUT_COST_PER_MILLION_TOKENS,
    QA_OUTPUT_COST_PER_MILLION_TOKENS,
)
from models import LLMReason
from store import get_screenplay_scenes, list_screenplays, record_llm_cost, search_chunks

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__, "0.1.0")

# Tool declarations for Gemini (OpenAPI-style parameters)
LIST_SCREENPLAYS_DECL = {
    "name": "list_screenplays",
    "description": "List all screenplays stored in the database. Returns id, title, authors, and source filename for each. Use this to find which screenplay(s) exist and their IDs before querying scenes or searching content.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

GET_SCENES_DECL = {
    "name": "get_screenplay_scenes",
    "description": "Get the list of scenes for a given screenplay. Returns scene_index, heading, page, location, and time_of_day for each scene. Use screenplay_id from list_screenplays.",
    "parameters": {
        "type": "object",
        "properties": {
            "screenplay_id": {
                "type": "string",
                "description": "The screenplay ID (UUID from list_screenplays).",
            },
        },
        "required": ["screenplay_id"],
    },
}

SEMANTIC_SEARCH_DECL = {
    "name": "semantic_search",
    "description": "Search screenplay content by meaning (semantic similarity). Use a natural language query (e.g. 'dialogue about the murder', 'scenes in the restaurant'). Returns matching chunk text, metadata (scene_heading, page, location, characters), and scene_index (0-based). Use scene_index in your [[ref:SCENE_INDEX \"quote\"]] citations. Optionally restrict to one screenplay by screenplay_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query describing what to find in the screenplay.",
            },
            "screenplay_id": {
                "type": "string",
                "description": "Optional. If provided, search only within this screenplay (UUID).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
            },
        },
        "required": ["query"],
    },
}

TOOL_DECLARATIONS = [LIST_SCREENPLAYS_DECL, GET_SCENES_DECL, SEMANTIC_SEARCH_DECL]


def _get_client() -> genai.Client | None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _get_usage_from_response(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a Gemini response. Returns (0, 0) if not found."""
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None) or 0
    out = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None) or 0
    return int(inp) if inp is not None else 0, int(out) if out is not None else 0


def _embed_query(client: genai.Client, query: str, output_dim: int = 1536) -> list[float]:
    """Embed a single query string for semantic search."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(output_dimensionality=output_dim),
    )
    if hasattr(result, "embeddings") and result.embeddings:
        return result.embeddings[0].values
    return []


_CITATION_RE = re.compile(r'\[\[ref:(\d+)(?:\s+"([^"]*)")?\]\]')


@dataclass
class SceneReference:
    """One inline citation extracted from the LLM answer."""
    scene_index: int
    quote: str = ""


@dataclass
class ToolCallEntry:
    """Record of a single function call made during the agent loop."""
    name: str
    args: dict[str, Any]
    round: int
    duration_ms: float = 0.0
    result_keys: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class QAResult:
    """Structured result from answer_question."""
    answer: str
    references: list[SceneReference] = field(default_factory=list)
    tool_rounds: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)


def parse_citations(text: str) -> list[SceneReference]:
    """Extract [[ref:N "quote"]] citations from text."""
    refs: list[SceneReference] = []
    seen: set[tuple[int, str]] = set()
    for m in _CITATION_RE.finditer(text):
        scene_index = int(m.group(1))
        quote = m.group(2) or ""
        key = (scene_index, quote)
        if key not in seen:
            seen.add(key)
            refs.append(SceneReference(scene_index=scene_index, quote=quote))
    return refs


# Type for embedding function: list of texts -> list of vectors
EmbedFn = Callable[[list[str]], list[list[float]]]


async def _execute_tool(
    name: str,
    args: dict[str, Any],
    session: AsyncSession,
    embed_fn: EmbedFn | None,
    client: genai.Client | None = None,
    screenplay_id: str | None = None,
) -> dict[str, Any]:
    """Execute a tool by name and return a JSON-serializable result."""
    if name == "list_screenplays":
        items = await list_screenplays(session)
        return {"screenplays": items}

    if name == "get_screenplay_scenes":
        sid = args.get("screenplay_id")
        if sid is None:
            return {"error": "screenplay_id is required"}
        scenes = await get_screenplay_scenes(session, sid)
        return {"scenes": scenes}

    if name == "semantic_search":
        query = args.get("query") or ""
        search_screenplay_id = args.get("screenplay_id") or screenplay_id
        limit = args.get("limit", 10)
        if not query:
            return {"error": "query is required", "results": []}
        if not embed_fn:
            return {
                "error": "Semantic search requires embeddings. Re-ingest screenplays with GOOGLE_API_KEY set.",
                "results": [],
            }
        if client is not None:
            emb_result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=query,
                config=types.EmbedContentConfig(output_dimensionality=1536),
            )
            inp_tok, out_tok = _get_usage_from_response(emb_result)
            if inp_tok or out_tok:
                cost = (inp_tok + out_tok) / 1_000_000 * EMBEDDING_COST_PER_MILLION_TOKENS
                await record_llm_cost(
                    session,
                    LLMReason.EMBED_QUERY,
                    inp_tok,
                    out_tok,
                    screenplay_id=search_screenplay_id,
                    cost=cost,
                )
            query_embedding = (
                emb_result.embeddings[0].values
                if emb_result.embeddings
                else embed_fn([query])[0]
            )
        else:
            query_embedding = embed_fn([query])[0]
        results = await search_chunks(
            session,
            query_embedding,
            screenplay_id=search_screenplay_id if search_screenplay_id is not None else None,
            limit=min(int(limit), 20) if limit is not None else 10,
        )
        return {
            "results": [
                {"text": text, "metadata": meta, "scene_index": scene_index}
                for text, meta, scene_index in results
            ],
        }

    return {"error": f"Unknown tool: {name}"}


async def answer_question(
    session: AsyncSession,
    question: str,
    *,
    screenplay_id: str | None = None,
    embed_fn: EmbedFn | None = None,
    max_rounds: int = 10,
) -> QAResult:
    """
    Answer a question about the screenplay(s) using Gemini with function calling.

    If screenplay_id is provided, it is added to the system context so the model
    can prefer that screenplay. embed_fn is used for semantic_search; if None,
    semantic search is disabled (list_screenplays and get_screenplay_scenes still work).

    Returns a QAResult with the answer text (containing citation markers), parsed references,
    and telemetry about the agent loop (rounds, tokens, tool call log).
    """
    with tracer.start_as_current_span("qa.answer_question") as root_span:
        root_span.set_attribute("qa.question", question)
        if screenplay_id:
            root_span.set_attribute("qa.screenplay_id", screenplay_id)
        root_span.set_attribute("qa.max_rounds", max_rounds)

        client = _get_client()
        if not client:
            root_span.set_attribute("qa.error", "missing_api_key")
            return QAResult(answer="Error: Set GOOGLE_API_KEY or GEMINI_API_KEY to use the Q&A API.")

        tools = types.Tool(function_declarations=TOOL_DECLARATIONS)
        system_instruction = (
            "You are a helpful assistant that answers questions about screenplays. "
            "Use the provided tools to list screenplays, get their scenes, or search the content by meaning. "
            "When answering a question (e.g. about a character's age, relationships, or events), use semantic_search with a relevant query to retrieve screenplay chunks. "
            "The search returns actual text from the screenplay—read that text and answer from it. Extract or infer details from the retrieved content when the question asks for them; if the content does not mention what is asked, say so. "
            "Answer based only on the tool results; if no relevant data is found, say so.\n\n"
            "CITATIONS: When your answer references specific text, events, or dialogue from the screenplay, "
            "you MUST include inline citations so the reader can jump to the source. Use this exact format:\n"
            '  [[ref:SCENE_INDEX "short exact quote"]]\n'
            "Rules:\n"
            "- SCENE_INDEX is the 0-based scene_index from semantic_search results.\n"
            "- The quote must be a short exact phrase (roughly 5-15 words) copied from the screenplay text.\n"
            "- Place each citation immediately after the claim or detail it supports.\n"
            "- Include citations for every specific factual claim drawn from the screenplay.\n"
            "- You may use multiple citations in one answer.\n"
            '- Example: John reveals he is a retired detective [[ref:5 "I hung up my badge ten years ago"]].\n'
        )
        if screenplay_id is not None:
            system_instruction += f" The user may be asking about screenplay ID {screenplay_id} in particular."

        config = types.GenerateContentConfig(
            tools=[tools],
            temperature=0.2,
            system_instruction=system_instruction,
        )

        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=question)],
            ),
        ]

        total_inp = 0
        total_out = 0
        tool_call_log: list[dict[str, Any]] = []

        def _build_result(answer: str, rounds: int) -> QAResult:
            refs = parse_citations(answer)
            root_span.set_attribute("qa.rounds", rounds)
            root_span.set_attribute("qa.total_input_tokens", total_inp)
            root_span.set_attribute("qa.total_output_tokens", total_out)
            root_span.set_attribute("qa.tool_call_count", len(tool_call_log))
            root_span.set_attribute("qa.reference_count", len(refs))
            root_span.set_attribute("qa.answer_length", len(answer))
            logger.info(
                "QA complete: rounds=%d tool_calls=%d input_tokens=%d output_tokens=%d refs=%d answer_len=%d",
                rounds, len(tool_call_log), total_inp, total_out, len(refs), len(answer),
            )
            return QAResult(
                answer=answer,
                references=refs,
                tool_rounds=rounds,
                total_input_tokens=total_inp,
                total_output_tokens=total_out,
                tool_call_log=tool_call_log,
            )

        for round_idx in range(max_rounds):
            with tracer.start_as_current_span(f"qa.round.{round_idx}") as round_span:
                round_span.set_attribute("qa.round_index", round_idx)

                with tracer.start_as_current_span("qa.generate_content") as gen_span:
                    t0 = time.monotonic()
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=config,
                    )
                    gen_duration_ms = (time.monotonic() - t0) * 1000
                    gen_span.set_attribute("qa.generate_duration_ms", gen_duration_ms)

                inp_tokens, out_tokens = _get_usage_from_response(response)
                total_inp += inp_tokens
                total_out += out_tokens
                round_span.set_attribute("qa.input_tokens", inp_tokens)
                round_span.set_attribute("qa.output_tokens", out_tokens)
                logger.info(
                    "QA round %d: generate_content took %.0fms, input_tokens=%d output_tokens=%d",
                    round_idx, gen_duration_ms, inp_tokens, out_tokens,
                )

                if inp_tokens or out_tokens:
                    cost = (
                        inp_tokens / 1_000_000 * QA_INPUT_COST_PER_MILLION_TOKENS
                        + out_tokens / 1_000_000 * QA_OUTPUT_COST_PER_MILLION_TOKENS
                    )
                    await record_llm_cost(
                        session,
                        LLMReason.QA_ANSWER,
                        inp_tokens,
                        out_tokens,
                        screenplay_id=screenplay_id,
                        cost=cost,
                    )

                if not response.candidates:
                    root_span.set_attribute("qa.error", "no_candidates")
                    return _build_result("No response from the model.", round_idx + 1)

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    root_span.set_attribute("qa.error", "empty_response")
                    return _build_result("Empty response from the model.", round_idx + 1)

                parts = list(candidate.content.parts)
                function_calls = [
                    p for p in parts
                    if hasattr(p, "function_call") and p.function_call is not None
                ]
                text_parts = [p for p in parts if hasattr(p, "text") and p.text]

                round_span.set_attribute("qa.function_call_count", len(function_calls))
                round_span.set_attribute("qa.has_text", bool(text_parts))

                if text_parts and not function_calls:
                    answer_text = text_parts[0].text.strip()
                    return _build_result(answer_text, round_idx + 1)

                if not function_calls:
                    answer_text = text_parts[0].text.strip() if text_parts else "No answer generated."
                    return _build_result(answer_text, round_idx + 1)

                logger.info(
                    "QA round %d: model requested %d function call(s): %s",
                    round_idx,
                    len(function_calls),
                    ", ".join(p.function_call.name for p in function_calls),
                )

                contents.append(
                    types.Content(role="model", parts=parts),
                )

                response_parts = []
                for call_idx, part in enumerate(function_calls):
                    fc = part.function_call
                    name = fc.name
                    try:
                        args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                    except Exception:
                        args = {}

                    with tracer.start_as_current_span(f"qa.tool.{name}") as tool_span:
                        tool_span.set_attribute("qa.tool.name", name)
                        tool_span.set_attribute("qa.tool.args", json.dumps(args, default=str))
                        tool_span.set_attribute("qa.tool.round", round_idx)
                        tool_span.set_attribute("qa.tool.call_index", call_idx)

                        logger.info(
                            "QA round %d: calling tool %s with args %s",
                            round_idx, name, json.dumps(args, default=str),
                        )

                        t0 = time.monotonic()
                        try:
                            result = await _execute_tool(
                                name, args, session, embed_fn,
                                client=client,
                                screenplay_id=screenplay_id,
                            )
                        except Exception as exc:
                            tool_duration_ms = (time.monotonic() - t0) * 1000
                            tool_span.record_exception(exc)
                            tool_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                            logger.error(
                                "QA round %d: tool %s failed after %.0fms: %s",
                                round_idx, name, tool_duration_ms, exc,
                            )
                            result = {"error": str(exc)}

                        tool_duration_ms = (time.monotonic() - t0) * 1000
                        tool_span.set_attribute("qa.tool.duration_ms", tool_duration_ms)
                        tool_span.set_attribute("qa.tool.result_keys", list(result.keys()))

                        if "error" in result:
                            tool_span.set_attribute("qa.tool.error", result["error"])

                        result_summary = _summarize_tool_result(name, result)
                        tool_span.set_attribute("qa.tool.result_summary", result_summary)
                        logger.info(
                            "QA round %d: tool %s completed in %.0fms — %s",
                            round_idx, name, tool_duration_ms, result_summary,
                        )

                        tool_call_log.append({
                            "name": name,
                            "args": args,
                            "round": round_idx,
                            "duration_ms": round(tool_duration_ms, 1),
                            "result_summary": result_summary,
                        })

                    response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response=result,
                        ),
                    )

                contents.append(
                    types.Content(
                        role="user",
                        parts=response_parts,
                    ),
                )

        root_span.set_attribute("qa.error", "max_rounds_exceeded")
        return _build_result("Maximum tool-calling rounds reached. Try a simpler question.", max_rounds)


def _summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    """Produce a concise human-readable summary of a tool result for logs/traces."""
    if "error" in result:
        return f"error: {result['error']}"
    if name == "list_screenplays":
        items = result.get("screenplays", [])
        return f"{len(items)} screenplay(s)"
    if name == "get_screenplay_scenes":
        scenes = result.get("scenes", [])
        return f"{len(scenes)} scene(s)"
    if name == "semantic_search":
        results = result.get("results", [])
        return f"{len(results)} chunk(s) returned"
    return f"keys={list(result.keys())}"
