import torch
import numpy as np
from typing import Dict

"""
Feature extractor that converts audio waveforms into token IDs.
It currently reshapes the audio waveform into a tensor and returns it as a dictionary with the key "input_values".
Likely has to do with huggingface architectural requirements when training the model.
"""
class HiggsAudioFeatureExtractor(torch.nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt") -> Dict[str, torch.Tensor]:
        # Convert from librosa to torch
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}
