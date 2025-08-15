import os
import sys

import torch
from transformers import LlamaConfig

# Ensure project root is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.mor_model.modeling_llama import MoRLlamaForCausalLM
from model.kv_caches.cache_utils import DynamicCache


def tiny_config():
    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        vocab_size=100,
        max_position_embeddings=32,
    )


def test_mor_llama_forward_cpu():
    config = tiny_config()
    model = MoRLlamaForCausalLM(config).to("cpu")
    input_ids = torch.ones((1, 4), dtype=torch.long, device="cpu")
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (1, 4, config.vocab_size)
    assert outputs.logits.device.type == "cpu"


def test_dynamic_cache_update():
    cache = DynamicCache()
    key = torch.zeros(1, 1, 1, 1)
    value = torch.zeros(1, 1, 1, 1)
    cache.update(key, value, layer_idx=0)
    assert cache.get_seq_length(0) == 1
