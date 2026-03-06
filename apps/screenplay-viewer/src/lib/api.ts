import type { LLMCostsResponse, SceneReference, Screenplay, ScreenplayListItem } from "@/types/screenplay";

const API_BASE =
  typeof window !== "undefined"
    ? (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000")
    : process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function listScreenplays(): Promise<ScreenplayListItem[]> {
  const res = await fetch(`${API_BASE}/screenplays`);
  if (!res.ok) throw new Error("Failed to list screenplays");
  return res.json();
}

export async function getScreenplay(id: string): Promise<Screenplay> {
  const res = await fetch(`${API_BASE}/screenplays/${id}`);
  if (!res.ok) {
    if (res.status === 404) throw new Error("Screenplay not found");
    throw new Error("Failed to load screenplay");
  }
  return res.json();
}

export async function deleteScreenplay(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/screenplays/${id}`, { method: "DELETE" });
  if (!res.ok) {
    if (res.status === 404) throw new Error("Screenplay not found");
    throw new Error("Failed to delete screenplay");
  }
}

export async function updateScreenplayMetadata(
  id: string,
  patch: { title?: string; authors?: string[] },
): Promise<void> {
  const res = await fetch(`${API_BASE}/screenplays/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    if (res.status === 404) throw new Error("Screenplay not found");
    throw new Error("Failed to update screenplay");
  }
}

export async function updateSceneMetadata(
  screenplayId: string,
  sceneIndex: number,
  patch: { location_type?: string; location?: string; time_of_day?: string },
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/screenplays/${screenplayId}/scenes/${sceneIndex}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  );
  if (!res.ok) {
    if (res.status === 404) throw new Error("Scene not found");
    throw new Error("Failed to update scene");
  }
}

export async function updateElementText(
  screenplayId: string,
  sceneIndex: number,
  elementIndex: number,
  text: string,
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/screenplays/${screenplayId}/scenes/${sceneIndex}/elements/${elementIndex}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    },
  );
  if (!res.ok) {
    if (res.status === 404) throw new Error("Element not found");
    throw new Error("Failed to update element");
  }
}

export interface AskResponse {
  answer: string;
  references: SceneReference[];
}

export async function getLLMCosts(): Promise<LLMCostsResponse> {
  const res = await fetch(`${API_BASE}/costs`);
  if (!res.ok) throw new Error("Failed to fetch LLM costs");
  return res.json();
}

export async function askQuestion(
  question: string,
  screenplayId: string,
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      screenplay_id: screenplayId,
    }),
  });
  if (!res.ok) throw new Error("Failed to get answer");
  return res.json();
}

/** Optional generation params for Higgs TTS (passed as query params). */
export interface GenerateAudioOptions {
  speaker_description?: string;
  scene_description?: string;
  temperature?: number;
  seed?: number;
  max_tokens?: number;
  force_audio_gen?: boolean;
}

/**
 * Fetch stored audio for a scene (by screenplay_id + scene_index).
 * Returns the audio blob. Throws with 404 if no audio has been generated yet.
 */
export async function getSceneAudio(
  screenplayId: string,
  sceneIndex: number,
): Promise<Blob> {
  const res = await fetch(
    `${API_BASE}/screenplays/${screenplayId}/scenes/${sceneIndex}/audio`,
  );
  if (!res.ok) {
    if (res.status === 404) throw new Error("NOT_FOUND");
    let message = "Failed to fetch audio";
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return res.blob();
}

/**
 * Request generation of audio for a single scene via higgs-tts.
 * Saves to DB and returns the audio blob for playback.
 */
export async function generateSceneAudio(
  screenplayId: string,
  sceneIndex: number,
  options?: GenerateAudioOptions,
): Promise<Blob> {
  const url = new URL(
    `${API_BASE}/screenplays/${screenplayId}/scenes/${sceneIndex}/generate`,
  );
  if (options) {
    if (options.speaker_description != null)
      url.searchParams.set("speaker_description", options.speaker_description);
    if (options.scene_description != null)
      url.searchParams.set("scene_description", options.scene_description);
    if (options.temperature != null)
      url.searchParams.set("temperature", String(options.temperature));
    if (options.seed != null) url.searchParams.set("seed", String(options.seed));
    if (options.max_tokens != null)
      url.searchParams.set("max_tokens", String(options.max_tokens));
    if (options.force_audio_gen != null)
      url.searchParams.set("force_audio_gen", String(options.force_audio_gen));
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    let message = "Failed to generate audio";
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
    } catch {
      if (res.status === 404) message = "Screenplay or scene not found";
      if (res.status === 503) message = "TTS service unavailable. Is higgs-tts running?";
    }
    throw new Error(message);
  }
  return res.blob();
}

/**
 * Request generation of audio for the first scene via higgs-tts.
 * Returns the audio blob for playback.
 */
export async function generateFirstSceneAudio(
  screenplayId: string,
  options?: GenerateAudioOptions,
): Promise<Blob> {
  const url = new URL(`${API_BASE}/screenplays/${screenplayId}/generate`);
  if (options) {
    if (options.speaker_description != null) url.searchParams.set("speaker_description", options.speaker_description);
    if (options.scene_description != null) url.searchParams.set("scene_description", options.scene_description);
    if (options.temperature != null) url.searchParams.set("temperature", String(options.temperature));
    if (options.seed != null) url.searchParams.set("seed", String(options.seed));
    if (options.max_tokens != null) url.searchParams.set("max_tokens", String(options.max_tokens));
    if (options.force_audio_gen != null) url.searchParams.set("force_audio_gen", String(options.force_audio_gen));
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    let message = "Failed to generate audio";
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
    } catch {
      if (res.status === 404) message = "Screenplay not found";
      if (res.status === 503) message = "TTS service unavailable. Is higgs-tts running?";
    }
    throw new Error(message);
  }
  return res.blob();
}
