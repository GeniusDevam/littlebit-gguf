# Train LittleBit-2 on Kaggle

This run adapts Qwen3-0.6B at a target of 0.55 bits per projection weight. Each projection has a primary and a residual path. Both use the same rank, rounded to the nearest multiple of 128 with a minimum of 128. K and V use the same rank formula as the other projections. The token embeddings, tied output head and normalization weights are frozen.

The trainer clones [mgiachetti/little-bit](https://github.com/mgiachetti/little-bit/tree/02d146a72551762b68cb250577c6f42c29cf65a0) (MIT) and loads its LittleBit-2 Joint-ITQ initializer and SmoothSign gradient. The adapter uses the checkpoint's operation order, rounds scales to their on-disk BF16 precision, and exports the upstream rank scale as one BF16 vector multiplied by a vector of ones. `littlebit_config.json` records `use_itq: true`. GPU matmuls use FP16 with a loss scaler. The teacher is frozen and the student uses gradient checkpointing.

## Training recipe

The default follows the pinned upstream Qwen3-0.6B run D. It averages teacher-to-student KL over all token positions and gives it weight 1. The hidden-state error has weight 10. That error omits position zero, where the attention sink produces an unusually large activation. It divides each layer's MSE by the teacher's mean squared activation at the remaining positions, then averages across layers. The trainer calls the upstream loss directly. [Attention-sink ablation](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#L65-L97) [Loss and defaults](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/train.py#L35-L102)

Adam uses a peak learning rate of `8e-5`, linear warmup for the first 2% of the planned steps (at least one step), cosine decay toward zero, and gradient clipping at 1.0. The trainer calls the same upstream schedule using this run's cumulative target as its horizon. The default target is 100 million input tokens. [Run D launch arguments](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/scripts/qwen_c4_1b.sh#L6-L11) [Schedule](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/train.py#L71-L102) [Adam and clipping](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/train.py#L161-L209)

Each update consumes two 2,048-token sequences. The trainer averages their gradients and keeps one sequence's activations live at a time. Run D processes both sequences together, so its relative MSE uses one teacher-energy denominator per layer across the batch. Accumulation uses one per sequence, as upstream run C does. Gradient checkpointing and 128-token logit chunks also follow run C. This trainer uses FP16 on T4, export-aligned ranks, BF16-rounded scales and frozen normalization weights. Run D used TF32, trained norms and had a longer schedule and corpus. This trainer mixes the full pinned C4 shard with four copies of WikiText-2 train. [Run C and D settings](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#L103-L116) [Run C command](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#L175-L184)

To run the previous recipe in a fresh run, set `LEGACY_RECIPE = True` in the notebook or use `--legacy-recipe`. This selects equal weights for KL and absolute hidden-state MSE, a constant `3e-5` learning rate, one sequence per step, an 8,388,608-token C4 prefix, one copy of WikiText-2 train and a 2,097,152-token target. The token cache and progress logging are available with both recipes.

## Start a run

1. Make a Kaggle account. Open your account settings and complete **Phone Verification** to unlock accelerators and Internet. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
2. Create a notebook. Use **File > Import Notebook** to open [littlebit_qat.ipynb](littlebit_qat.ipynb).
3. Open the notebook settings, turn **Internet** on, and select **GPU** as the accelerator. Choose a T4 or P100. If Kaggle shows two T4s, the notebook uses the first GPU. The setup uses the PyTorch CUDA 12.6 wheel for Pascal compatibility. [PyTorch CUDA wheels](https://docs.pytorch.org/get-started/previous-versions/)
4. Leave `RUN_TRAIN = True`, with `RUN_GGUF` and `UPLOAD` off. Start the new recipe with `RESUME` empty. Set `TOKENS` before the first session if you want a target other than 100 million. The trainer consumes whole optimizer steps, so a target can round down by up to one step. `C4_TOKENS` can limit the shard for a short trial. The default tokenizes the whole shard.
5. Use **Save Version > Save & Run All**. The 630-minute deadline starts before notebook setup. Training stops between optimizer steps, reserving the rest of Kaggle's session for saving, export and full WikiText-2 test perplexity for the teacher and student. The notebook logs a student perplexity baseline and repeats the check every 30 minutes on up to the first four fixed WikiText-2 test windows. Compare these progress readings across sessions. The full-test result covers the entire split.
6. Open the finished version and select **Output**. Under `littlebit-train/` you will find `tokens.pt` (the tokenized train and test corpora), `resume.pt`, `littlebit/` (the checkpoint and tokenizer), and `summary.json`. The summary includes the training commit, packed projection bits per weight, throughput, stop reason, peak allocated GPU memory and full-test perplexity.

### T4 session plan

The first Kaggle run at `73b7a20` used one Tesla T4 and processed 2,097,152 tokens in 3,087 seconds (51.45 minutes) including setup and final evaluation. Its last reported training rate was 821.10 tokens/s. Peak allocated GPU memory was 6,488,476,672 bytes (6.49 GB). Full WikiText-2 test perplexity was 20.96 for the teacher and 1,067.20 for the student.

The following figures are arithmetic from the measured training rate. At 821.10 tokens/s, 630 minutes of uninterrupted training would process 31.04 million tokens. Setup, cache transfer or creation, progress checks, and export consume time. The rate and memory use of the new recipe still need a T4 measurement.

| Cumulative target | Training hours at 821.10 tokens/s | Sessions at the 630-minute cutoff |
| --- | ---: | ---: |
| 25 million | 8.46 | 1 |
| 50 million | 16.92 | 2 |
| 100 million | 33.83 | 4 |

Kaggle documents a 12-hour CPU and GPU session limit and describes its weekly GPU quota as floating. Using 30 GPU hours as a planning reference, 25M and 50M fit within a week at the measured rate. The 100M target requires more than 30 hours before overhead, so plan for time across two quota windows if your account shows 30 hours. Check the quota shown in the notebook before starting another session. [Session duration](https://www.kaggle.com/product-feedback/302908) [Floating weekly quota](https://www.kaggle.com/discussions/product-feedback/173129)

## Continue in another session

Use **Add Input** to attach the previous notebook version's output. Set `RESUME` to the attached `littlebit-train/resume.pt`. Keep `RESULTS` empty for training. The trainer picks up `tokens.pt` beside the resume file automatically. To point at a separate cache or reuse it in a fresh run, set `TOKEN_CACHE` (CLI `--token-cache`). Each session writes a copy of the cache to its output, so attaching the latest output is enough. A cache must match the tokenizer and dataset revisions and C4 limit. Set the same recipe and training settings, including the cumulative target, in each continuation. Change `MINUTES` to fit the time available and `LOGIT_CHUNK` to fit memory.

The trainer replaces `resume.pt` before each 30-minute progress check and saves again when training stops. It restores the latent factors, optimizer, loss scaler, random state, and completed step. The next step reads the next two sequences in the same fixed shuffle and evaluates the same cumulative learning-rate schedule. The default recipe fixes the planned step count at the beginning. The legacy constant-rate recipe allows a larger target on resume.

The notebook reads the exact source commit from the attached `summary.json`. If you recovered a standalone `resume.pt`, set `TRAINING_COMMIT` from the original logs. The attached input is read-only. The new version writes its own output under `/kaggle/working`.

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

Importing the upstream trainer also loads its data module, which uses Hugging Face Datasets (Apache-2.0). This trainer reads the pinned corpora directly. [Hugging Face Datasets license](https://github.com/huggingface/datasets/blob/main/LICENSE)

The datasets permit research use. C4's database license covers the collection. Individual web pages may carry their own terms. Training mixes complete sequences from the two train sources after shuffling with a fixed seed. The test split is held out. [LittleBit paper](https://arxiv.org/abs/2506.13771) [LittleBit-2 paper](https://proceedings.mlr.press/v306/lee26y.html)
