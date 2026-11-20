import gzip
import json
from array import array
from itertools import islice
from pathlib import Path

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


def training_blocks(
    tokenizer: PreTrainedTokenizerBase, seq_len: int, c4_limit: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    wiki = (
        tokenizer(
            wikitext("train"),
            add_special_tokens=False,
            return_tensors="pt",
            verbose=False,
        )
        .input_ids.flatten()
        .int()
    )
    c4 = c4_tokens(tokenizer, c4_limit)
    blocks = torch.cat(
        [tokens[: len(tokens) // seq_len * seq_len] for tokens in (wiki, c4)]
    )
    blocks = blocks.view(-1, seq_len)
    if not len(blocks):
        raise ValueError("training data contains no complete sequences")
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed))
    print(
        f"Corpus: WikiText-2 {len(wiki)} tokens, C4 {len(c4)} tokens, {len(blocks)} blocks"
    )
    return blocks, order


def write_wiki_test(output: Path) -> None:
    output.write_text(wikitext("test"), encoding="utf-8")
