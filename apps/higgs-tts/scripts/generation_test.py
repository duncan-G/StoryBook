"""Example script for generating audio using HiggsAudio directly."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.engine import AudioEngine
from src.data_models.generation_input import GenerationInput
from src.data_models.message import Message
from src.data_models.message_content import TextContent
from src.data_models.speaker import Speaker


# ============================================================================
# Speakers
# ============================================================================

ALEX = Speaker(
    description=(
        "Male, American accent, modern speaking rate, moderate-pitch, "
        "friendly tone, and very clear audio."
    ),
)

# ============================================================================
# Messages
# ============================================================================

MESSAGES = [
    Message(
        role="user",
        content=TextContent(
            text=(
                "Hey, everyone! Welcome back to Tech Talk Tuesdays. "
                "It's your host, Alex, and today, we're diving into a topic "
                "that's become absolutely crucial in the tech world - deep learning. "
                "And let's be honest, if you've been even remotely connected to tech, "
                "AI, or machine learning lately, you know that deep learning is everywhere."
            ),
        ),
        speaker=ALEX,
    ),
]

SCENE_DESCRIPTION = "Audio is recorded from a quiet room."

# Generation parameters
MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.95
RAS_WIN_LEN = 7
RAS_WIN_MAX_NUM_REPEAT = 2
SEED = 123

OUTPUT_DIR = Path(__file__).resolve().parent / "audio_outputs"

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"


# ============================================================================
# Main
# ============================================================================

def main():
    print("Initializing AudioEngine...")
    engine = AudioEngine(
        model_name_or_path=MODEL_PATH,
        tokenizer_name_or_path=MODEL_PATH,
        audio_tokenizer_name_or_path=AUDIO_TOKENIZER_PATH,
    )

    inputs = [
        GenerationInput(
            messages=MESSAGES,
            scene_description=SCENE_DESCRIPTION,
        ),
    ]

    print("Generating audio...")
    audio_chunks = []

    for idx, gen_input in enumerate(inputs):
        print(f"Generating input {idx + 1}/{len(inputs)}...")
        response = engine.generate(
            chat=gen_input,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            ras_win_len=RAS_WIN_LEN,
            ras_win_max_num_repeat=RAS_WIN_MAX_NUM_REPEAT,
            seed=SEED,
        )

        if response.audio is not None:
            audio_chunks.append(response.audio)
            print(f"  Generated {len(response.audio) / response.sampling_rate:.2f} seconds of audio")
        else:
            print(f"  Warning: No audio generated for input {idx + 1}")

    if not audio_chunks:
        print("Error: No audio was generated!")
        return

    print("Concatenating audio chunks...")
    final_audio = np.concatenate(audio_chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{timestamp}.wav"

    print(f"Saving audio to {output_path}...")
    sampling_rate = response.sampling_rate if response.sampling_rate else 24000
    sf.write(str(output_path), final_audio, sampling_rate)

    total_duration = len(final_audio) / sampling_rate
    print(f"Audio saved successfully!")
    print(f"  Total duration: {total_duration:.2f} seconds")
    print(f"  Sampling rate: {sampling_rate} Hz")


if __name__ == "__main__":
    main()
