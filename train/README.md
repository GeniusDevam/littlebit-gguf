# Train LittleBit-2 on Kaggle

This run adapts Qwen3-0.6B at a target of 0.55 bits per projection weight. Each projection has a primary and a residual path. Both use the same rank, rounded to the nearest multiple of 128 with a minimum of 128. K and V use the same rank formula as the other projections. The token embeddings, tied output head and normalization weights are frozen.

The trainer clones [mgiachetti/little-bit](https://github.com/mgiachetti/little-bit/tree/02d146a72551762b68cb250577c6f42c29cf65a0) (MIT) and loads its LittleBit-2 Joint-ITQ initializer and SmoothSign gradient. The adapter uses the checkpoint's operation order, rounds scales to their on-disk BF16 precision, and exports the upstream rank scale as one BF16 vector multiplied by a vector of ones. `littlebit_config.json` records `use_itq: true`. GPU matmuls use FP16 with a loss scaler. The teacher is frozen and the student uses gradient checkpointing.

## Start a run

1. Make a Kaggle account. Open your account settings and complete **Phone Verification** to unlock accelerators and Internet. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
2. Create a notebook. Use **File > Import Notebook** to open [littlebit_qat.ipynb](littlebit_qat.ipynb).
3. Open the notebook settings, turn **Internet** on, and select **GPU** as the accelerator. Choose a T4 or P100. If Kaggle shows two T4s, the notebook uses the first GPU. The setup uses the PyTorch CUDA 12.6 wheel for Pascal compatibility. [PyTorch CUDA wheels](https://docs.pytorch.org/get-started/previous-versions/)
4. Leave `RUN_TRAIN = True`, with `RUN_GGUF` and `UPLOAD` off. `TOKENS` is the cumulative input-token budget. Training stops at the last complete sequence within the budget. `SEQ_LEN` is the number of tokens per sequence. `C4_TOKENS` limits how much of the selected shard is tokenized. Set it to `0` to use the full shard.
5. Use **Save Version > Save & Run All**. Check the version's logs for setup progress and the training lines with loss, tokens per second, and token-budget ETA. The time budget starts before notebook setup and defaults to 630 minutes. Training stops at an optimizer boundary and uses the remaining session time for checkpointing, export and test perplexity.
6. Open the finished version and select **Output**. Under `littlebit-train/` you will find `resume.pt`, `littlebit/` (the checkpoint and tokenizer), and `summary.json`. The summary includes the training commit, ranks, packed projection bits per weight, throughput, stop reason, peak allocated GPU memory and WikiText-2 test perplexity for both models.

The first logged step includes warm-up. Use the rate after several steps to estimate the remaining training time. Downloading C4, initializing the factors, and evaluating the full test split add time beyond the logged training ETA. On an Intel Core i7-8550U in Ubuntu WSL2 with two threads, the tiny CPU notebook processed 64 random input tokens at 110.8 tokens/s and finished in 17.2 seconds including setup and evaluation. That fixture has a small vocabulary and two layers. For a GPU planning reference, the pinned upstream report's Qwen3-0.6B run C processed 26,000 steps of 4,096 tokens in 66.5 hours. At that rate, the default token budget would take about 1.31 hours of training. At half that rate it would take 2.62 hours. Allow additional time for setup and test evaluation, and use your own session's logged rate to revise the estimate. [Upstream training report](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#33-scaling-the-training-data)

## Continue in another session

Use **Add Input** to attach the previous notebook version's output. Find `littlebit-train/resume.pt` in the input file browser and copy its path into `RESUME`. Leave `RESULTS` empty for training. `TOKENS` includes progress from earlier sessions, so increase it if the previous session already met its budget.

The trainer replaces `resume.pt` about every 30 minutes and saves again when training stops. It restores the latent factors, optimizer, loss scaler, random state, and step, then continues with the next sequence in the same deterministic data order. Keep the seed, sequence length, learning rate and C4 limit the same. Change `MINUTES`, `TOKENS` and `LOGIT_CHUNK` to fit the next session. The attached input is read-only. The new version writes its own output under `/kaggle/working`.

The notebook reads the exact source commit from the attached `summary.json`. If you recovered a standalone `resume.pt`, set `TRAINING_COMMIT` from the original logs.

## Convert and evaluate

Run from the repository root on Linux or WSL with CMake, Ninja, a C++ compiler and the repository's Python requirements installed.

```sh
bash scripts/build.sh
python convert_littlebit_to_gguf.py /path/to/littlebit-train/littlebit \
    --outfile littlebit-q8_0.gguf --embd-type q8_0 --validate 2
python convert_littlebit_to_gguf.py /path/to/littlebit-train/littlebit \
    --outfile littlebit-q4_k.gguf --embd-type q4_k --validate 2
build/bin/llama-cli -m littlebit-q4_k.gguf -ngl 0 -t 4 -c 512 \
    --single-turn -p 'Once upon a time' -n 24
```

For the notebook comparison, start a separate CPU session with Internet on. Attach the training output, set `RESULTS` to the attached `littlebit-train` directory, turn `RUN_TRAIN` off and turn `RUN_GGUF` on. If needed, install CMake, Ninja and a C++ compiler in the session before running the evaluation cell. The notebook checks out the training commit, builds the patched llama.cpp and converts the pinned teacher to dense BF16. It evaluates all three GGUF files with `llama-perplexity` on the same WikiText-2 test text and runs a short `llama-cli` sample. Find the CPU details, converter output, file sizes, perplexity and sample in `littlebit-evaluation/evaluation.log`.

For an optional private upload, set `HF_REPO`, add `HF_TOKEN` to Kaggle Secrets, grant the notebook access, and turn `UPLOAD` on. The cell uploads only `littlebit/`.

## Local checks

The existing CPU build gives the tests access to the llama.cpp logit reader. Use a clean checkout of the upstream revision above, set `LITTLEBIT_UPSTREAM` to its path, and install `train/requirements.txt` plus pytest.

```sh
python -m train.train_qat --plan
python -m pytest -s tests/test_train.py
python -m train.train_qat --tiny --cpu --seq-len 16 --tokens 64 --output /tmp/littlebit-tiny
```

Use an empty output directory for each run. `--plan` fetches the base config and prints ranks and projection sizes without model weights. For a local notebook smoke run, set `LITTLEBIT_TINY=1`, `LITTLEBIT_REPO` to this checkout, and `LITTLEBIT_OUTPUT` to an empty directory. Use a kernel with the training requirements installed. Set `LITTLEBIT_UPSTREAM` if the pinned layer code is already available locally.

## Model and data licenses

| Asset | Revision | License |
| --- | --- | --- |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 |
| [allenai/c4](https://huggingface.co/datasets/allenai/c4/tree/1588ec454efa1a09f29cd18ddd04fe05fc8653a2), English train shard 00000 | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | ODC-BY for the database |
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3), `wikitext-2-raw-v1` train and test | `b08601e04326c79dfdd32d625aee71d232d685c3` | CC BY-SA 3.0 and GFDL |

The datasets permit research use. C4's database license covers the collection. Individual web pages may carry their own terms. Training mixes complete sequences from the two train sources after shuffling with a fixed seed. The test split is held out. [LittleBit paper](https://arxiv.org/abs/2506.13771) [LittleBit-2 paper](https://proceedings.mlr.press/v306/lee26y.html)
