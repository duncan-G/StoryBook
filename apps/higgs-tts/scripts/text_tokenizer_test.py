#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:05:10 2025

"""

from transformers import AutoTokenizer

"""
    https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/tree/main
    Contains tokenizer weights and additional metadata.
    See tokenizer.json, tokenizer_config.json, special_tokens_map.json, config.json
    
    Inspecting the tokenizer, we can see the following config:

        Vocab Size: 128,000
        Model max length: 131,072
        Is Fast: true
        Padding side: right
        Truncation side: right
        Special tokens: {'bos_token': '<|begin_of_text|>', 'eos_token': '<|end_of_text|>'}
        clean_up_tokenization_spaces: true

    Higgs tokenizer is a BPE tokenizer (https://huggingface.co/learn/llm-course/en/chapter6/5)
        
"""
MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
MODEL_CACHE_DIR = "../.models"

def load_model():
    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR)

if __name__ == '__main__':
    tokenizer = load_model()
    
    text = "Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model. It’s used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa."
    
    # Tokenize will split text into individual strings.
    # add_special_tokens=True will add the special tokens to the beginning of the text.
    # Note: Ġ character is used to represent the string starts with a space
    tokens = tokenizer.tokenize(text, add_special_tokens=True)
    print(tokens)

    # Convert tokens into token ids.
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)