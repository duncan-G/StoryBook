"""Simple encode and decode smoke test for the Higgs audio tokenizer.
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Allow running the script directly without installing the repo as a package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_tokenizer.higgs_audio_tokenizer import HiggsAudioTokenizer


BASE_DIR = Path(__file__).resolve().parent
# Hardcoded input file provided by the user, relative to this script directory
INPUT_WAV = BASE_DIR / "voice_prompts" / "belinda.wav"

# Point this to your tokenizer (env override preferred)
TOKENIZER_PATH = os.environ.get("HIGGS_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer")

# Choose device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if not INPUT_WAV.exists():
        raise FileNotFoundError(f"Input WAV not found: {INPUT_WAV}")

    if not TOKENIZER_PATH:
        raise ValueError(
            "Set HIGGS_TOKENIZER_PATH environment variable to a local tokenizer directory or a Hugging Face repo id."
        )

    output_dir = INPUT_WAV.parent / "../audio_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{INPUT_WAV.stem}_decoded.wav"

    # Delete the output file if it already exists
    if output_path.exists():
        output_path.unlink()

    tokenizer = HiggsAudioTokenizer.load(
        tokenizer_name_or_path=TOKENIZER_PATH,
        device=DEVICE
    )

    # Encode to codes
    codes = tokenizer.encode(str(INPUT_WAV))
    
    # Ensure codes have batch dimension (decode expects 3D tensor: B, N, T or B, T, N)
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)

    # Decode back to waveform
    decoded = tokenizer.decode(codes)  # numpy array, shape (B, C, T)
    audio = np.squeeze(decoded)
    if audio.ndim == 2:
        # (C, T) -> (T, C) for soundfile
        audio = audio.T

    # Clamp to valid audio range to avoid clipping when saving
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(output_path, audio, samplerate=tokenizer.sampling_rate)
    print(f"Encoded and decoded '{INPUT_WAV.name}'. Saved to '{output_path}'.")


if __name__ == "__main__":
    main()
