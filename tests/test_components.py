import os
import sys

import torch
from transformers import LlamaConfig, AutoTokenizer
from datasets import Dataset

# Ensure project root is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.base_model.modeling_llama import LlamaForCausalLM as BaseLlamaForCausalLM
from model.recursive_model.modeling_llama import LlamaForCausalLM as RecursiveLlamaForCausalLM
from lm_dataset.language_modeling_dataset import LanguageModelingDataset


def tiny_config():
    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        vocab_size=100,
        max_position_embeddings=32,
    )


def run_forward(model_cls):
    config = tiny_config()
    model = model_cls(config).to("cpu")
    input_ids = torch.ones((1, 4), dtype=torch.long, device="cpu")
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (1, 4, config.vocab_size)
    assert outputs.logits.device.type == "cpu"


def test_base_llama_forward():
    run_forward(BaseLlamaForCausalLM)


def test_recursive_llama_forward():
    run_forward(RecursiveLlamaForCausalLM)


def test_language_modeling_dataset_iterates():
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.padding_side = "right"
    ds = Dataset.from_dict({"text": ["hello world"]})
    lm_ds = LanguageModelingDataset(
        ds,
        tokenizer=tokenizer,
        max_length=4,
        continuous=False,
        buffer_size=32,
        global_shuffling=False,
        local_shuffling=False,
    )
    sample = next(iter(lm_ds))
    assert "input_ids" in sample and sample["input_ids"].dtype == torch.long
