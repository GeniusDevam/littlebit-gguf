import gzip
import json
import shutil
import tempfile
from array import array
from itertools import islice
from pathlib import Path
from typing import Callable

import torch
from huggingface_hub import hf_hub_download
from pyarrow.parquet import read_table
from transformers import PreTrainedTokenizerBase

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
C4 = "allenai/c4"
C4_REV = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_SHARD = "en/c4-train.00000-of-01024.json.gz"
WIKI = "Salesforce/wikitext"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"


def wikitext(split: str) -> str:
    path = hf_hub_download(
        WIKI,
        f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=WIKI_REV,
        token=False,
    )
    return "\n\n".join(read_table(path, columns=["text"])["text"].to_pylist())


def c4_tokens(tokenizer: PreTrainedTokenizerBase, limit: int) -> torch.Tensor:
    path = hf_hub_download(
        C4, C4_SHARD, repo_type="dataset", revision=C4_REV, token=False
    )
    separator = tokenizer("\n\n", add_special_tokens=False).input_ids
    tokens = array("i")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        while documents := [json.loads(row)["text"] for row in islice(source, 128)]:
            ids = tokenizer(
                documents, add_special_tokens=False, verbose=False
            ).input_ids
            for document in ids:
                tokens.extend(document)
                tokens.extend(separator)
            if limit and len(tokens) >= limit:
                del tokens[limit:]
                break
    return torch.frombuffer(tokens, dtype=torch.int32)


def tokenize_corpus(
    tokenizer: PreTrainedTokenizerBase, c4_limit: int
) -> dict[str, torch.Tensor]:
    tokens = {}
    for split in ("train", "test"):
        tokens[split] = (
            tokenizer(
                wikitext(split),
                add_special_tokens=False,
                return_tensors="pt",
                verbose=False,
            )
            .input_ids.flatten()
            .int()
        )
    tokens["c4"] = c4_tokens(tokenizer, c4_limit)
    return tokens


def token_cache(
    destination: Path,
    source: Path | None,
    signature: dict,
    tokenize: Callable[[], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    def read(path: Path) -> dict[str, torch.Tensor]:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        tokens = state["tokens"]
        if (
            state["signature"] != signature
            or tokens.keys() != {"train", "test", "c4"}
            or any(ids.ndim != 1 or ids.dtype != torch.int32 for ids in tokens.values())
            or len(tokens["test"]) < 2
        ):
            raise ValueError(f"token cache does not match the corpus settings: {path}")
        return tokens

    if source is not None and source.resolve() != destination.resolve():
        if destination.exists():
            raise ValueError(
                f"remove the existing cache before using --token-cache: {destination}"
            )
        with tempfile.TemporaryDirectory(
            prefix=".tokens-", dir=destination.parent
        ) as temp:
            pending = Path(temp) / "tokens.pt"
            shutil.copyfile(source, pending)
            tokens = read(pending)
            pending.replace(destination)
        print(f"Reused token cache from {source}", flush=True)
        return tokens
    if not destination.exists():
        if source is not None:
            raise FileNotFoundError(source)
        print("Tokenizing training and test data", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=".tokens-", dir=destination.parent
        ) as temp:
            pending = Path(temp) / "tokens.pt"
            torch.save({"signature": signature, "tokens": tokenize()}, pending)
            pending.replace(destination)
        print(f"Saved token cache to {destination}", flush=True)
    else:
        print(f"Reused token cache from {destination}", flush=True)
    return read(destination)


def training_blocks(
    tokens: dict[str, torch.Tensor], seq_len: int, seed: int, wiki_repeat: int
) -> tuple[torch.Tensor, torch.Tensor]:
    wiki, c4 = tokens["train"], tokens["c4"]
    wiki_blocks = wiki[: len(wiki) // seq_len * seq_len]
    c4_blocks = c4[: len(c4) // seq_len * seq_len]
    blocks = torch.cat([wiki_blocks] * wiki_repeat + [c4_blocks]).view(-1, seq_len)
    if not len(blocks):
        raise ValueError("training data contains no complete sequences")
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed))
    print(
        f"Corpus: WikiText-2 {len(wiki)} tokens x {wiki_repeat}, "
        f"C4 {len(c4)} tokens, {len(blocks)} blocks",
        flush=True,
    )
    return blocks, order


def write_wiki_test(output: Path) -> None:
    output.write_text(wikitext("test"), encoding="utf-8")
