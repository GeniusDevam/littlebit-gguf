import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config, Qwen3Config

from littlebit import field_name, prefixes


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


def make_tiny_littlebit(
    path: Path, ranks: tuple[int, ...], residual: bool = True, width: int = 256
) -> None:
    """Write a small LittleBit checkpoint with fixed ranks and numpy-seeded weights."""
    make_tiny_model(path, width=width)
    weights = path / "model.safetensors"
    state = load_file(weights)
    rng = np.random.default_rng(3)
    tensors = {}
    names = set()
    for index, prefix in enumerate(prefixes(2)):
        rows, cols = state[f"{prefix}.weight"].shape
        rank = ranks[index % len(ranks)]
        for extra in range(1 + residual):
            fields = {
                "U_shape": torch.tensor([rows, rank]),
                "V_shape": torch.tensor([rank, cols]),
                "U_packed": rng.integers(-(2**31), 2**31, (rows, -(-rank // 32))),
                "V_packed": rng.integers(-(2**31), 2**31, (rank, cols // 32)),
            }
            for field, size in (("u1", rows), ("u2", rank), ("v1", rank), ("v2", cols)):
                fields[field] = rng.uniform(0.02, 0.2, (1, size))
            for field, value in fields.items():
                if not isinstance(value, torch.Tensor):
                    dtype = torch.int32 if "packed" in field else torch.bfloat16
                    value = torch.from_numpy(value).to(dtype)
                tensors[field_name(prefix, field, bool(extra))] = value
        names.add(f"{prefix}.weight")
    for name, tensor in state.items():
        if name not in names:
            noise = rng.standard_normal(tensor.shape) * 0.05
            tensors[name] = torch.from_numpy(noise).to(tensor.dtype)
    save_file(tensors, weights)
    (path / "littlebit_config.json").write_text(
        json.dumps({"eff_bit": 0.5, "residual": residual, "use_itq": False}),
        encoding="utf-8",
    )
