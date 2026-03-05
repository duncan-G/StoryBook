# Audio tokens
AUDIO_IN_TOKEN = "<|AUDIO|>"
AUDIO_OUT_TOKEN = "<|AUDIO_OUT|>"
AUDIO_BOS = "<|audio_bos|>"
AUDIO_EOS = "<|audio_eos|>"
AUDIO_OUT_BOS = "<|audio_out_bos|>"
# Alias for backward compatibility in generation context
AUDIO_PLACEHOLDER_TOKEN = AUDIO_IN_TOKEN

# Text formatting tokens
BEGIN_OF_TEXT = "<|begin_of_text|>"
START_HEADER_ID = "<|start_header_id|>"
END_HEADER_ID = "<|end_header_id|>"
RECIPIENT = "<|recipient|>"

# Scene description tokens
SCENE_DESC_START = "<|scene_desc_start|>"
SCENE_DESC_END = "<|scene_desc_end|>"

# Termination tokens
EOT_ID = "<|eot_id|>"  # End of turn
EOM_ID = "<|eom_id|>"  # End of message

# Legacy
EOS_TOKEN = "<|end_of_text|>"

# Default system messages
DEFAULT_SYSTEM_MESSAGE = "Generate audio following instruction."
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = "Generate audio following instruction."

# Whisper processor, 30 sec -> 3000 features
# Then we divide 4 in the audio tokenizer, we decrease 3000 features to 750, which gives 25 Hz
WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC = 25
