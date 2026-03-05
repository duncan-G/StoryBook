import type { Screenplay } from "@/types/screenplay";

/** Gemini 2.5 Flash Preview pricing (per million tokens) */
const INPUT_COST_PER_MILLION = 0.5;
const OUTPUT_COST_PER_MILLION = 10;

/** Screenplay standard: 1 page = 1 minute of screen time */
const MINUTES_PER_PAGE = 1;

/** ~25 output tokens ≈ 1 second of generated audio */
const OUTPUT_TOKENS_PER_SECOND = 25;

/** Approximate characters per token for English dialogue */
const CHARS_PER_TOKEN = 4;

export interface DialogueAudioEstimate {
  /** Total pages in screenplay (1 page = 1 minute) */
  totalPages: number;
  /** Estimated minutes of dialogue (from pages) */
  estimatedMinutes: number;
  /** Dialogue character count used for input */
  dialogueCharacters: number;
  /** Estimated input tokens (dialogue text sent to model) */
  inputTokens: number;
  /** Estimated output tokens (from audio length) */
  outputTokens: number;
  /** Estimated cost in USD */
  costUsd: number;
}

/**
 * Collect all dialogue text from screenplay (DIALOGUE and VOICE_OVER elements)
 * for token estimation.
 */
function getDialogueTextLength(screenplay: Screenplay): number {
  let chars = 0;
  for (const scene of screenplay.scenes) {
    for (const el of scene.elements) {
      if (el.type === "DIALOGUE" || el.type === "VOICE_OVER") {
        if (el.character) chars += el.character.length + 1;
        chars += el.text.length + 1;
      }
    }
  }
  return chars;
}

/**
 * Get the maximum page number in the screenplay (screenplay length in pages).
 */
function getTotalPages(screenplay: Screenplay): number {
  let maxPage = 0;
  for (const scene of screenplay.scenes) {
    for (const el of scene.elements) {
      if (el.page > maxPage) maxPage = el.page;
    }
  }
  return maxPage;
}

/**
 * Estimate cost for generating dialogue audio using Gemini 2.5 Flash Preview.
 * Uses screenplay standard: 1 page = 1 minute for duration; estimates input
 * tokens from dialogue text and output tokens from estimated minutes.
 */
export function estimateDialogueAudioCost(screenplay: Screenplay): DialogueAudioEstimate {
  const dialogueCharacters = getDialogueTextLength(screenplay);
  const totalPages = getTotalPages(screenplay) || 1;
  const estimatedMinutes = totalPages * MINUTES_PER_PAGE;
  const estimatedSeconds = estimatedMinutes * 60;
  const inputTokens = Math.ceil(dialogueCharacters / CHARS_PER_TOKEN);
  const outputTokens = estimatedSeconds * OUTPUT_TOKENS_PER_SECOND;
  const costUsd =
    (inputTokens / 1_000_000) * INPUT_COST_PER_MILLION +
    (outputTokens / 1_000_000) * OUTPUT_COST_PER_MILLION;

  return {
    totalPages,
    estimatedMinutes,
    dialogueCharacters,
    inputTokens,
    outputTokens,
    costUsd,
  };
}
