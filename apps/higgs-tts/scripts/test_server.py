"""gRPC client that sends a Generate request to the inference server and
writes the returned audio to a WAV file.

Usage:
    python scripts/test_server.py
    python scripts/test_server.py --host localhost --port 50051 --output output.wav
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_grpc import inference_engine_pb2 as pb2
from inference_grpc import inference_engine_pb2_grpc as pb2_grpc


# ============================================================================
# Speakers
# ============================================================================

ALEX = pb2.Speaker(
    name="SPEAKER0",
    description=(
        "Male, American accent, modern speaking rate, moderate-pitch, "
        "friendly tone, and very clear audio."
    ),
)

# ============================================================================
# Messages  (single-speaker example — swap / add speakers for multi-speaker)
# ============================================================================

MESSAGES = [
    pb2.InputMessage(
        text=(
            "Hey, everyone! Welcome back to Tech Talk Tuesdays. "
            "It's your host, Alex, and today, we're diving into a topic "
            "that's become absolutely crucial in the tech world - deep learning. "
            "<SE>[Laughter]</SE> I know, I know, you've probably heard that phrase "
            "a thousand times by now. But seriously, this stuff is fascinating. "
            "<SE>[Applause]</SE> Thank you, thank you! "
            "Alright, let's get into it."
        ),
        speaker=ALEX,
    ),
]

SCENE_DESCRIPTION = "Audio is recorded from a quiet room."


# ============================================================================
# Main
# ============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "audio_outputs"


def main():
    parser = argparse.ArgumentParser(description="gRPC audio generation client")
    parser.add_argument("--host", default=os.getenv("HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "50051")))
    parser.add_argument("--output", default=None, help="Output path (default: audio_outputs/<timestamp>.wav)")
    args = parser.parse_args()

    address = f"{args.host}:{args.port}"
    print(f"Connecting to {address}...")

    channel = grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    stub = pb2_grpc.InferenceEngineStub(channel)

    request = pb2.GenerateRequest(
        request_id=str(uuid.uuid4()),
        inputs=[
            pb2.GenerationInput(
                messages=MESSAGES,
                scene_description=SCENE_DESCRIPTION,
            ),
        ],
        sampling_params=pb2.SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=50,
            max_tokens=2048,
            seed=123,
            ras_win_len=7,
            ras_win_max_num_repeat=2,
        ),
        stream=False,
    )

    print("Sending Generate request...")
    audio_data = b""
    sampling_rate = 0
    audio_format = "flac"

    for response in stub.Generate(request):
        resp_type = response.WhichOneof("response")

        if resp_type == "chunk":
            chunk = response.chunk
            print(f"  Stream chunk: {chunk.completion_tokens} tokens")

        elif resp_type == "complete":
            complete = response.complete
            print(f"  Complete: finish_reason={complete.finish_reason}, "
                  f"prompt_tokens={complete.prompt_tokens}, "
                  f"completion_tokens={complete.completion_tokens}")
            if complete.audio_data:
                audio_data = complete.audio_data
                sampling_rate = complete.sampling_rate
                audio_format = complete.audio_format or audio_format

    channel.close()

    if not audio_data:
        print("Error: No audio received from server!")
        return

    ext = audio_format.lower()
    if args.output is not None:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_path = OUTPUT_DIR / f"{timestamp}.{ext}"

    output_path.write_bytes(audio_data)

    import soundfile as sf
    info = sf.info(output_path)
    print(f"Audio saved to {output_path}")
    print(f"  Format: {audio_format.upper()}, Duration: {info.duration:.2f}s, Sampling rate: {sampling_rate} Hz")


if __name__ == "__main__":
    main()
