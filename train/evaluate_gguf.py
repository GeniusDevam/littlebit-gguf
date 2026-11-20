import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from train.data import MODEL, MODEL_REV, write_wiki_test

ROOT = Path(__file__).resolve().parents[1]


def evaluate(results: Path, output: Path, threads: int) -> None:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if commit != summary["training_commit"]:
        raise ValueError(
            f"check out training commit {summary['training_commit']} first"
        )
    output.mkdir(parents=True, exist_ok=False)
    llama = ROOT / "third_party" / "llama.cpp"
    labels = ("q8_0", "q4_k")
    dense = output / "qwen3-bf16.gguf"
    wiki_test = output / "wikitext-2-test.txt"

    with (
        tempfile.TemporaryDirectory(prefix="littlebit-build-") as build_dir,
        (output / "evaluation.log").open("w", encoding="utf-8") as log,
    ):
        build = Path(build_dir)
        binaries = build / "bin"

        def run(*args: str | Path, env: dict[str, str] | None = None) -> None:
            print(" ".join(map(str, args)), flush=True)
            print(" ".join(map(str, args)), file=log, flush=True)
            subprocess.run(
                args,
                cwd=ROOT,
                env=env,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        run("lscpu")
        run("git", "rev-parse", "HEAD")
        run(
            "bash",
            ROOT / "scripts" / "build.sh",
            env=os.environ | {"LITTLEBIT_BUILD_DIR": str(build), "LITTLEBIT_JOBS": "2"},
        )
        for label in labels:
            run(
                sys.executable,
                ROOT / "convert_littlebit_to_gguf.py",
                results / "littlebit",
                "--outfile",
                output / f"littlebit-{label}.gguf",
                "--embd-type",
                label,
                "--validate",
                "2",
                "--binary-dir",
                binaries,
            )

        dense_source = snapshot_download(
            MODEL,
            revision=MODEL_REV,
            token=False,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
        )
        run(
            sys.executable,
            llama / "convert_hf_to_gguf.py",
            dense_source,
            "--outtype",
            "bf16",
            "--outfile",
            dense,
        )
        write_wiki_test(wiki_test)
        for gguf in [dense, *(output / f"littlebit-{kind}.gguf" for kind in labels)]:
            print(f"{gguf.name}: {gguf.stat().st_size} bytes", file=log, flush=True)
            run(
                binaries / "llama-perplexity",
                "-m",
                gguf,
                "-f",
                wiki_test,
                "-ngl",
                "0",
                "-t",
                str(threads),
                "-c",
                "2048",
            )
        run(
            binaries / "llama-cli",
            "-m",
            output / "littlebit-q4_k.gguf",
            "-ngl",
            "0",
            "-t",
            str(threads),
            "-c",
            "512",
            "--single-turn",
            "--simple-io",
            "--no-warmup",
            "-p",
            "Once upon a time",
            "-n",
            "24",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LittleBit GGUF on the CPU.")
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    evaluate(args.results.resolve(), args.output.resolve(), args.threads)


if __name__ == "__main__":
    main()
