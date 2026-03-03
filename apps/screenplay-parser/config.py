"""
App config; env overrides where applicable.
"""
from __future__ import annotations

import os

# Embedding API cost: $ per 1M tokens (e.g. Gemini embedding pricing).
EMBEDDING_COST_PER_MILLION_TOKENS: float = float(
    os.getenv("EMBEDDING_COST_PER_MILLION_TOKENS", "0.15")
)

# Q&A (generate_content) model: $ per 1M input / output tokens (e.g. Gemini 2.5 Flash / Flash-Lite).
QA_INPUT_COST_PER_MILLION_TOKENS: float = float(
    os.getenv("QA_INPUT_COST_PER_MILLION_TOKENS", "0.10")
)
QA_OUTPUT_COST_PER_MILLION_TOKENS: float = float(
    os.getenv("QA_OUTPUT_COST_PER_MILLION_TOKENS", "0.40")
)
