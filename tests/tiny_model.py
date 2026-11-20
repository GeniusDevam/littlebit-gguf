import json
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config, Qwen3Config


def make_tiny_model(path: Path, width: int = 256, architecture: str = "llama") -> None:
    path.mkdir()
    text = "The quick brown fox jumps over the lazy dog. Packing binary weights is fun."
    SentencePieceTrainer.train(
        sentence_iterator=iter([text, text.lower()]),
        model_prefix=str(path / "tokenizer"),
        vocab_size=64,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizer", "model_max_length": 256}),
        encoding="utf-8",
    )
    config_class = {"llama": LlamaConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}[
        architecture
    ]
    vocabulary = SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
    config = config_class(
        vocab_size=len(vocabulary),
        hidden_size=width,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=width // 4,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(7)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
