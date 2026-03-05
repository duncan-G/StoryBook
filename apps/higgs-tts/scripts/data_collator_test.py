#!/usr/bin/env python3
"""
Smoke test for the data collator pipeline.

This test:
1. Creates sample Chat objects (with text and optionally audio)
2. Converts them to HiggsAudioModelInput using InputProcessor
3. Runs them through HiggsAudioDataCollator to create a batched ModelBatchInput
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, WhisperProcessor
from typing import List

# Allow running the script directly without installing the repo as a package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.data_models.generation_input import GenerationInput
from src.data_models.message import Message
from src.data_models.message_content import TextContent, AudioContent
from src.input_processor import InputProcessor
from src.data_collator.higgs_audio_data_collator import HiggsAudioDataCollator
from src.audio_tokenizer.higgs_audio_tokenizer import HiggsAudioTokenizer


# Configuration
MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_ID = "bosonai/higgs-audio-v2-tokenizer"
MODEL_CACHE_DIR = ".models"
VOICE_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "voice_prompts")

# Default token IDs (from config.py)
AUDIO_IN_TOKEN_ID = 128015
AUDIO_OUT_TOKEN_ID = 128016


def create_sample_inputs() -> List[GenerationInput]:
    """Create sample GenerationInput objects for testing with multimodal text-to-speech examples."""
    chats = []
    
    # Get available audio files - use absolute paths
    audio_files = {
        "belinda": os.path.join(VOICE_PROMPTS_DIR, "belinda.wav"),
        "bigbang_amy": os.path.join(VOICE_PROMPTS_DIR, "bigbang_amy.wav"),
        "en_man": os.path.join(VOICE_PROMPTS_DIR, "en_man.wav"),
        "en_woman": os.path.join(VOICE_PROMPTS_DIR, "en_woman.wav"),
    }
    
    # Verify audio files exist
    available_audio = {k: v for k, v in audio_files.items() if os.path.exists(v)}
    if not available_audio:
        print(f"   Warning: No audio files found in {VOICE_PROMPTS_DIR}")
        print(f"   Falling back to text-only conversations")
    
    # Chat 1: Text-to-Speech - User asks for audio, assistant generates audio
    # This demonstrates the core TTS functionality
    if "en_woman" in available_audio:
        chat1 = GenerationInput(messages=[
            Message(
                role="user",
                content=TextContent(text="Can you say 'Hello, how are you today?' in a friendly voice?")
            ),
            Message(
                role="assistant",
                content=AudioContent(
                    audio_url=available_audio["en_woman"],
                    raw_audio=None
                )
            ),
        ])
        chats.append(chat1)
    
    # Chat 2: Speech-to-Text + Text-to-Speech - User provides audio, assistant responds with audio
    # This demonstrates bidirectional multimodal conversation
    if "belinda" in available_audio and "en_man" in available_audio:
        chat2 = GenerationInput(messages=[
            Message(
                role="user",
                content=AudioContent(
                    audio_url=available_audio["belinda"],
                    raw_audio=None
                )
            ),
            Message(
                role="assistant",
                content=TextContent(text="I heard your message. Let me respond with audio.")
            ),
            Message(
                role="assistant",
                content=AudioContent(
                    audio_url=available_audio["en_man"],
                    raw_audio=None
                )
            ),
        ])
        chats.append(chat2)
    
    # Chat 3: Mixed conversation - Text and audio in the same conversation
    if "bigbang_amy" in available_audio:
        chat3 = GenerationInput(messages=[
            Message(
                role="user",
                content=TextContent(text="What is the capital of France?")
            ),
            Message(
                role="assistant",
                content=TextContent(text="The capital of France is Paris.")
            ),
            Message(
                role="user",
                content=TextContent(text="Can you say that in an audio message?")
            ),
            Message(
                role="assistant",
                content=AudioContent(
                    audio_url=available_audio["bigbang_amy"],
                    raw_audio=None
                )
            ),
        ])
        chats.append(chat3)
    
    # Chat 4: Simple text conversation (no audio) - baseline test
    chat4 = GenerationInput(messages=[
        Message(
            role="user",
            content=TextContent(text="Hello, how are you?")
        ),
        Message(
            role="assistant",
            content=TextContent(text="I'm doing well, thank you for asking!")
        ),
    ])
    chats.append(chat4)
    
    return chats


def load_tokenizers():
    """Load text and audio tokenizers."""
    print("Loading text tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR
    )
    
    print("Loading audio tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tokenizer = HiggsAudioTokenizer.load(
        tokenizer_name_or_path=AUDIO_TOKENIZER_ID,
        device=device
    )
    
    print("Loading Whisper processor...")
    whisper_processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        cache_dir=MODEL_CACHE_DIR
    )
    
    return text_tokenizer, audio_tokenizer, whisper_processor, device


def get_token_ids(text_tokenizer: AutoTokenizer) -> dict:
    """Get special token IDs from the tokenizer."""
    # Get token IDs from tokenizer vocabulary
    vocab = text_tokenizer.get_vocab()
    
    # Try to get audio token IDs
    audio_in_token_id = vocab.get("<|AUDIO|>", AUDIO_IN_TOKEN_ID)
    audio_out_token_id = vocab.get("<|AUDIO_OUT|>", AUDIO_OUT_TOKEN_ID)
    
    # Get pad token ID
    pad_token_id = text_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = text_tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0  # Fallback
    
    # Audio stream BOS/EOS - use defaults that match common configs
    # These are typically not in the main vocab but used internally
    audio_stream_bos_id = vocab.get("<|audio_stream_bos|>", 100)
    audio_stream_eos_id = vocab.get("<|audio_stream_eos|>", 101)
    
    return {
        "audio_in_token_id": audio_in_token_id,
        "audio_out_token_id": audio_out_token_id,
        "pad_token_id": pad_token_id,
        "audio_stream_bos_id": audio_stream_bos_id,
        "audio_stream_eos_id": audio_stream_eos_id,
    }


def main():
    """Run the smoke test."""
    print("=" * 60)
    print("Data Collator Smoke Test")
    print("=" * 60)
    
    # Step 1: Create sample chats
    print("\n1. Creating sample GenerationInput objects...")
    chats = create_sample_inputs()
    print(f"   Created {len(chats)} input(s)")

    audio_count = sum(
        1 for inp in chats
        for msg in inp.messages
        if isinstance(msg.content, AudioContent)
    )
    print(f"   Total audio messages: {audio_count}")
    print(f"   Voice prompts directory: {VOICE_PROMPTS_DIR}")
    
    # Step 2: Load tokenizers
    print("\n2. Loading tokenizers...")
    text_tokenizer, audio_tokenizer, whisper_processor, device = load_tokenizers()
    print(f"   Device: {device}")
    
    # Step 3: Get token IDs
    print("\n3. Getting special token IDs...")
    token_ids = get_token_ids(text_tokenizer)
    print(f"   Audio IN token ID: {token_ids['audio_in_token_id']}")
    print(f"   Audio OUT token ID: {token_ids['audio_out_token_id']}")
    print(f"   Pad token ID: {token_ids['pad_token_id']}")
    
    # Step 4: Create InputProcessor
    print("\n4. Creating InputProcessor...")
    input_processor = InputProcessor(
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        device=torch.device(device)
    )
    
    # Step 5: Convert chats to model inputs
    print("\n5. Converting chats to HiggsAudioModelInput...")
    try:
        model_inputs = input_processor.process_inputs(chats)
        print(f"   ✓ Converted {len(model_inputs)} chat(s) to model input(s)")
        
        for i, model_input in enumerate(model_inputs):
            audio_shape = model_input.audio_ids_concat.shape if model_input.audio_ids_concat is not None and model_input.audio_ids_concat.numel() > 0 else "None/Empty"
            print(f"   Chat {i+1}: input_ids shape={model_input.input_ids.shape}, "
                  f"label_ids shape={model_input.label_ids.shape}, "
                  f"audio_ids_concat={audio_shape}")
    except Exception as e:
        print(f"   ✗ Failed to convert chats: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 6: Create DataCollator
    print("\n6. Creating HiggsAudioDataCollator...")
    data_collator = HiggsAudioDataCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=token_ids["audio_in_token_id"],
        audio_out_token_id=token_ids["audio_out_token_id"],
        pad_token_id=token_ids["pad_token_id"],
        audio_stream_bos_id=token_ids["audio_stream_bos_id"],
        audio_stream_eos_id=token_ids["audio_stream_eos_id"],
        encode_whisper_embed=True,
        return_audio_in_tokens=True,
    )
    
    # Step 7: Run through data collator
    print("\n7. Running through data collator...")
    try:
        batch_input = data_collator(model_inputs)
        print("   ✓ Successfully created ModelBatchInput")
        print(f"   Batch input_ids shape: {batch_input.input_ids.shape}")
        print(f"   Batch attention_mask shape: {batch_input.attention_mask.shape}")
        if batch_input.audio_features is not None:
            print(f"   Audio features shape: {batch_input.audio_features.shape}")
            print(f"   Audio feature attention_mask shape: {batch_input.audio_feature_attention_mask.shape}")
        else:
            print(f"   Audio features: None (no audio in batch)")
        if batch_input.audio_in_ids is not None and batch_input.audio_in_ids.numel() > 0:
            print(f"   Audio IN ids shape: {batch_input.audio_in_ids.shape}")
            print(f"   Audio IN ids start shape: {batch_input.audio_in_ids_start.shape}")
        else:
            print(f"   Audio IN ids: None/Empty")
        if batch_input.audio_out_ids is not None and batch_input.audio_out_ids.numel() > 0:
            print(f"   Audio OUT ids shape: {batch_input.audio_out_ids.shape}")
            print(f"   Audio OUT ids start shape: {batch_input.audio_out_ids_start.shape}")
        else:
            print(f"   Audio OUT ids: None/Empty")
        if batch_input.label_ids is not None:
            print(f"   Label ids shape: {batch_input.label_ids.shape}")
        if batch_input.reward is not None and batch_input.reward.numel() > 0:
            print(f"   Reward shape: {batch_input.reward.shape}")
        
        print("\n" + "=" * 60)
        print("✓ Smoke test PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Smoke test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
