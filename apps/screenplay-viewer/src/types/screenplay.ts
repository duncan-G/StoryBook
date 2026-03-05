export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface SceneElement {
  type: string;
  text: string;
  page: number;
  bbox: BoundingBox;
  character: string | null;
}

export interface DialogueBlock {
  character: string;
  is_voice_over: boolean;
  speech: string;
  parentheticals: string[];
  lines: SceneElement[];
}

export interface Scene {
  heading: string;
  page: number;
  location_type: string | null;
  location: string;
  time_of_day: string;
  action_lines: string[];
  characters_present: string[];
  dialogue_blocks: DialogueBlock[];
  elements: SceneElement[];
}

export interface Screenplay {
  title: string;
  authors: string[];
  all_characters: string[];
  scenes: Scene[];
  elements: SceneElement[];
}

export interface ScreenplayListItem {
  id: string; // UUID v7
  title: string;
  authors: string[];
  source_filename: string | null;
}

export interface IngestResponse {
  screenplay_id: string; // UUID v7
  message: string;
  screenplay: Screenplay | null;
}

export interface SceneReference {
  scene_index: number;
  quote: string;
}

export interface LLMCostTotals {
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

export interface LLMCostByReason {
  reason: string;
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

export interface LLMCostByScreenplay {
  screenplay_id: string;
  title: string;
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

export interface LLMCostEntry {
  id: number;
  screenplay_id: string | null;
  screenplay_title: string | null;
  reason: string;
  input_tokens: number;
  output_tokens: number;
  cost: number | null;
  created_at: string | null;
}

export interface LLMCostsResponse {
  totals: LLMCostTotals;
  by_reason: LLMCostByReason[];
  by_screenplay: LLMCostByScreenplay[];
  recent: LLMCostEntry[];
}
