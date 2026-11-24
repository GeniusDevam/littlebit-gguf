# Train LittleBit-2 on Kaggle

This run adapts Qwen3-0.6B at a target of 0.55 bits per projection weight. Rounding the ranks and storing the scales puts the planned checkpoint at 0.618 bits per projection weight and the GGUF at 0.679 (`python -m train.train_qat --plan`). Each projection has a primary and a residual path. Both use the same rank, rounded to the nearest multiple of 128 with a minimum of 128. K and V use the same rank formula as the other projections. The token embeddings, tied output head and normalization weights are frozen.

The trainer clones [mgiachetti/little-bit](https://github.com/mgiachetti/little-bit/tree/02d146a72551762b68cb250577c6f42c29cf65a0) (MIT) and loads its LittleBit-2 Joint-ITQ initializer and SmoothSign gradient. The adapter uses the checkpoint's operation order, rounds scales to their on-disk BF16 precision, and exports the upstream rank scale as one BF16 vector multiplied by a vector of ones. `littlebit_config.json` records `use_itq: true`. GPU matmuls use FP16 with a loss scaler. The teacher is frozen and the student uses gradient checkpointing.

## Training recipe

The default follows the pinned upstream Qwen3-0.6B run D. It averages teacher-to-student KL over all token positions and gives it weight 1. The hidden-state error has weight 10. That error omits position zero, where the attention sink produces an unusually large activation. It divides each layer's MSE by the teacher's mean squared activation at the remaining positions, then averages across layers. The trainer calls the upstream loss directly. [Attention-sink ablation](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#L65-L97) [Loss and defaults](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/train.py#L35-L102)

Adam uses a peak learning rate of `8e-5`, linear warmup for the first 2% of the planned steps (at least one step), cosine decay toward zero, and gradient clipping at 1.0. The trainer calls the same upstream schedule using this run's cumulative target as its horizon. The default target is 100 million input tokens. [Run D launch arguments](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/scripts/qwen_c4_1b.sh#L6-L11) [Schedule](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/train.py#L71-L102) [Adam and clipping](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/train.py#L161-L209)

Each update consumes two 2,048-token sequences. The trainer averages their gradients and each GPU keeps one sequence's activations live at a time. Run D processes both sequences together, so its relative MSE uses one teacher-energy denominator per layer across the batch. Accumulation uses one per sequence, as upstream run C does. Gradient checkpointing and 128-token logit chunks also follow run C. This trainer uses FP16 on T4, export-aligned ranks, BF16-rounded scales and frozen normalization weights. Run D used TF32, trained norms and had a longer schedule and corpus. This trainer mixes the full pinned C4 shard with four copies of WikiText-2 train. [Run C and D settings](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#L103-L116) [Run C command](https://github.com/mgiachetti/little-bit/blob/02d146a72551762b68cb250577c6f42c29cf65a0/REPORT.md#L175-L184)

To run the previous recipe in a fresh run, set `LEGACY_RECIPE = True` and `GRAD_ACCUM = 2` in the notebook or use `--legacy-recipe`. The two GPUs need an even number of sequences per step, and the legacy default of one works only in a single process. The legacy recipe selects equal weights for KL and absolute hidden-state MSE, a constant `3e-5` learning rate, one sequence per step, an 8,388,608-token C4 prefix, one copy of WikiText-2 train and a 2,097,152-token target. The token cache and progress logging are available with both recipes.

## Two GPUs

A Kaggle run uses both T4s of the **GPU T4 x2** accelerator. This is the default and the only supported Kaggle mode. The notebook starts one process per GPU:

```sh
python -m torch.distributed.run --standalone --local-addr 127.0.0.1 --nproc-per-node 2 -m train.train_qat --output run
```

`--local-addr` keeps the launcher from looking up the host name, which can hang for minutes before the trainer's own limits start.

- **Split.** `--grad-accum` is the number of sequences per optimizer step and must be divisible by the number of processes (ranks). Rank r takes sequences `r, r + ranks, ...` of the step. Each rank scales its loss by `1 / grad_accum`, then the gradients are summed over the ranks before unscaling, clipping and the Adam update. With two sequences and two ranks this is the same single addition as accumulating both in one process, so the order of the data, the tokens per step and `resume.pt` are unchanged. With more sequences per rank the additions happen in another order and the results can differ in the last bits.
- **Same start.** Rank 0 initializes or restores the student and sends its parameters, buffers, optimizer state, loss scale and random state to the other rank. Rank 0 alone reads `resume.pt`, downloads, tokenizes, prints and writes files.
- **Agreement.** Rank 0 decides when to stop and when to check progress. A non-finite loss on either rank stops both. The ranks compare a checksum of every trained tensor at the start, after the first step of the session, on every step that is a multiple of `--log-every` (default 10) and before every save. They stop if one differs, so a drift shows up within a few steps and never replaces a good `resume.pt`. When the first step overflows in fp16 it changes no weights, and a later multiple covers the first real update.
- **Timeouts.** Waiting for rank 0 to download, tokenize or initialize allows 2 hours. Collectives during training allow 10 minutes. Joining the process group and the two all-reduce checks at the start allow 2 minutes each. When a rank fails, the launcher stops the other one.
- **Threads.** `--threads` is per process. The default of 2 uses the 4 CPU cores of a Kaggle session.

Look for these lines in the first minutes of a session:

| Log line | Meaning |
| --- | --- |
| `GPU 0: Tesla T4`, `GPU 1: Tesla T4` | The notebook found both GPUs. Otherwise it stops before installing anything. |
| `All-reduce check: 2 ranks` | The GPUs can talk to each other. If this times out, set `os.environ["NCCL_P2P_DISABLE"] = "1"` in the settings cell and start again. |
| `Gradient all-reduce: 1003577344 bytes in ...` | The time to sum one step's gradients (28 layers at the planned ranks). |
| `Starting at step ... 2 ranks Tesla T4, Tesla T4` | Both models are ready and the checkpoint is restored. |
| `first_step_check` | Rank 0 also computed both sequences alone. The run stops if the gradient norms differ by more than 1e-2 (relative), or if the gradients themselves differ by more than 1e-1 of their norm. An fp16 overflow moves the check to the next step. |
| `tokens_per_second` | Counts the tokens of all ranks. |

Without the launcher the trainer runs as one process on one device, which is meant for local checks.

## Start a run

1. Make a Kaggle account. Open your account settings and complete **Phone Verification** to unlock accelerators and Internet. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
2. Create a notebook. Use **File > Import Notebook** to open [littlebit_qat.ipynb](littlebit_qat.ipynb).
3. Open the notebook settings, turn **Internet** on, and select **GPU T4 x2** as the accelerator. The notebook stops before installing anything unless it sees exactly two GPUs. The setup uses the PyTorch CUDA 12.6 wheel. [PyTorch CUDA wheels](https://docs.pytorch.org/get-started/previous-versions/)
4. Leave `RUN_TRAIN = True`, with `RUN_GGUF` and `UPLOAD` off. Start the new recipe with `RESUME` empty. Set `TOKENS` before the first session if you want a target other than 100 million. The trainer consumes whole optimizer steps, so a target can round down by up to one step. `C4_TOKENS` can limit the shard for a short trial. The default tokenizes the whole shard.
5. Use **Save Version > Save & Run All**. The 630-minute deadline starts before notebook setup. Training stops between optimizer steps, reserving the rest of Kaggle's session for saving, export and the full WikiText-2 test and C4 validation perplexity for the teacher and student. The notebook logs a student perplexity baseline and repeats the check every 30 minutes on up to the first four fixed WikiText-2 test windows. Compare these progress readings across sessions. The full-test result covers the entire split.
6. Open the finished version and select **Output**. Under `littlebit-train/` you will find `tokens.pt` (the tokenized train and test corpora), `resume.pt`, `littlebit/` (the checkpoint and tokenizer), `c4-validation.txt` (the text behind the C4 score) and `summary.json`. The summary includes the training commit, packed projection bits per weight, throughput over all ranks, stop reason, the number of ranks, the GPU names, peak allocated memory per GPU and both perplexity scores.

### C4 score

Training reads WikiText-2 train four times and one C4 train shard, so WikiText-2 test is close to the training text. The final evaluation also scores the teacher and the student on text from the C4 *validation* split: the first 637 documents of `en/c4-validation.00000-of-00008.json.gz` at the pinned revision, each followed by a blank line. With the pinned tokenizer that is 300,509 tokens, against 299,078 in WikiText-2 test. The document count is fixed in `train/data.py` and logged. The summary reports `teacher_c4_validation_perplexity` and `student_c4_validation_perplexity`. Compare the student with the teacher on the same text. The two texts have different difficulty, so their scores are not comparable with each other.

The trainer tokenizes each document separately, as it does for the C4 training text. `llama-perplexity` tokenizes the joined file, which can merge text differently at document boundaries, and it scores only the second half of each window. Compare its results with each other rather than with the summary.

### Session plan

The first two sessions of the default recipe ran on one Tesla T4 at commit `3885d29`:

| Session | Steps | Tokens at the end | Rate | Student WikiText-2 test perplexity | Peak allocated GPU memory |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 0 to 6,917 | 28,332,032 | 763 tokens/s | 59.88 | 7.06 GB |
| 2 | 6,917 to 13,839 | 56,684,544 | 754 tokens/s | 46.11 | 7.07 GB |

The teacher scores 20.96. The 100M target is 24,414 steps, so 10,575 steps (43,315,200 tokens) remain.

| Remaining 43.3M tokens | Rate | Training hours |
| --- | ---: | ---: |
| One T4, measured rate of session 2 | 754 tokens/s | 15.95 |
| Both T4s at 1.6 times that rate (**estimate**) | 1,207 tokens/s | 9.97 |
| Both T4s at 1.9 times that rate (**estimate**) | 1,433 tokens/s | 8.40 |

The two-GPU rates are estimates until a Kaggle session measures them. Below about 1.55 times the single-GPU rate, the remaining steps no longer fit in the 630-minute cutoff minus about 10 minutes of setup and checks (an assumption based on the 205.5 s setup of session 2), and a fourth session finishes the run.

Kaggle documents a 12-hour CPU and GPU session limit and describes its weekly GPU quota as floating. Check the quota shown in the notebook before starting another session. [Session duration](https://www.kaggle.com/product-feedback/302908) [Floating weekly quota](https://www.kaggle.com/discussions/product-feedback/173129)

## Continue in another session

Use **Add Input** to attach the previous notebook version's output. Set `RESUME = "attached"` to use the single attached `littlebit-train/resume.pt`, or set `RESUME` to its path. The notebook stops when it finds none or several. Keep `RESULTS` empty for training. The trainer picks up `tokens.pt` beside the resume file automatically. To point at a separate cache or reuse it in a fresh run, set `TOKEN_CACHE` (CLI `--token-cache`). Each session writes a copy of the cache to its output, so attaching the latest output is enough. A cache must match the tokenizer and dataset revisions and C4 limit. Set the same recipe and training settings, including the cumulative target, in each continuation. Change `MINUTES` to fit the time available and `LOGIT_CHUNK` to fit memory.

The trainer replaces `resume.pt` before each 30-minute progress check and saves again when training stops. It restores the latent factors, optimizer, loss scaler, random state, and completed step. A checkpoint from a single-GPU session, including one written by `3885d29`, continues on two GPUs and the other way round. The next step reads the next two sequences in the same fixed shuffle and evaluates the same cumulative learning-rate schedule. The default recipe fixes the planned step count at the beginning. The legacy constant-rate recipe allows a larger target on resume.

A resumed run uses `TRAINING_COMMIT` if you set it and the `train-r3` branch otherwise. Evaluation with `RESULTS` still reads the exact source commit from the attached `summary.json`, so `RESUME` and `RESULTS` cannot both be set. The attached input is read-only. The new version writes its own output under `/kaggle/working`.

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

For the notebook comparison, start a separate CPU session with Internet on. Attach the training output, set `RESULTS` to the attached `littlebit-train` directory, turn `RUN_TRAIN` off and turn `RUN_GGUF` on. If needed, install CMake, Ninja and a C++ compiler in the session before running the evaluation cell. The notebook checks out the training commit, builds the patched llama.cpp and converts the pinned teacher to dense BF16. It evaluates all three GGUF files with `llama-perplexity` on the WikiText-2 test text and on `c4-validation.txt` from the results directory, and runs a short `llama-cli` sample. Find the CPU details, converter output, file sizes, perplexity and sample in `littlebit-evaluation/evaluation.log`.

For an optional private upload, set `HF_REPO`, add `HF_TOKEN` to Kaggle Secrets, grant the notebook access, and turn `UPLOAD` on. The cell uploads only `littlebit/`.

## Local checks

The existing CPU build gives the tests access to the llama.cpp logit reader. Use a clean checkout of the upstream revision above, set `LITTLEBIT_UPSTREAM` to its path, and install `train/requirements.txt` plus pytest.

```sh
python -m train.train_qat --plan
python -m pytest -s tests/test_train.py
python -m train.train_qat --tiny --cpu --seq-len 16 --tokens 64 --output /tmp/littlebit-tiny
```

The tests start two CPU processes over the `gloo` backend. Use an empty output directory for each run. `--plan` fetches the base config and prints ranks and projection sizes without model weights. For a local notebook smoke run, set `LITTLEBIT_TINY=1`, `LITTLEBIT_REPO` to this checkout, and `LITTLEBIT_OUTPUT` to an empty directory. Use a kernel with the training requirements installed. Set `LITTLEBIT_UPSTREAM` if the pinned layer code is already available locally.

## Model and data licenses

| Asset | Revision | License |
| --- | --- | --- |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 |
| [allenai/c4](https://huggingface.co/datasets/allenai/c4/tree/1588ec454efa1a09f29cd18ddd04fe05fc8653a2), English train shard 00000 and validation shard 00000 | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | ODC-BY for the database |
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3), `wikitext-2-raw-v1` train and test | `b08601e04326c79dfdd32d625aee71d232d685c3` | CC BY-SA 3.0 and GFDL |

Importing the upstream trainer also loads its data module, which uses Hugging Face Datasets (Apache-2.0). This trainer reads the pinned corpora directly. [Hugging Face Datasets license](https://github.com/huggingface/datasets/blob/main/LICENSE)

The datasets permit research use. C4's database license covers the collection. Individual web pages may carry their own terms. Training mixes complete sequences from the two train sources after shuffling with a fixed seed. The WikiText-2 test split and the C4 validation split are held out. [LittleBit paper](https://arxiv.org/abs/2506.13771) [LittleBit-2 paper](https://proceedings.mlr.press/v306/lee26y.html)
