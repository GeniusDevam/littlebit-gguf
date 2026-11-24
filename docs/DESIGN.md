# Design

## Goal

Convert a LittleBit / LittleBit-2 checkpoint (format in [FORMAT.md](FORMAT.md)) into a GGUF file whose compressed layers are stored as llama.cpp `Q1_0` tensors, and run it with a small patch to llama.cpp that adds no new ggml operations or kernels.

## Q1_0 as llama.cpp defines it

From `ggml/src/ggml-common.h` and `ggml-quants.c`:

- Block of 128 weights: `ggml_half d` followed by 16 bytes `qs`.
- Weight `j` of a block is bit `j % 8` of byte `j / 8` (least significant bit first).
- **Bit 1 means `+d`, bit 0 means `−d`.**
- A row's length must be a multiple of 128.

LittleBit stores **bit 0 as +1 and bit 1 as −1**, so the converter inverts every bit. The byte and bit order within a row are otherwise the same (LSB first), but LittleBit packs rows into 32-column words while `Q1_0` uses 128-column blocks with a scale in front, so rows are re-blocked.

## Mapping one layer

Per path (primary, and `_R` if present), with `l = v1 ⊙ u2`, `h = u1`, `g = v2`:

| GGUF tensor | Type | ggml shape `(ne0, ne1)` | Built from |
|---|---|---|---|
| `<base>.lb_g` | F32 | `(in)` | `g` |
| `<base>.lb_v` | Q1_0 | `(in, r_pad)`, or `(in, r)` in format version 2 | `V±` with bits inverted; every block in row `k` has `d = l_k` |
| `<base>.lb_u` | Q1_0 | `(r_pad, out)` | `U±` with bits inverted; every block in row `i` has `d = h_i` |

The residual path uses `lb_g_r`, `lb_v_r`, `lb_u_r`. `<base>` is llama.cpp's own tensor name (for example `blk.0.attn_q`, `blk.3.ffn_down`). The usual `<base>.weight` is absent. Biases keep their usual names (`blk.N.attn_q.bias`).

The layer then computes, in ggml:

```
t = ggml_mul(x, lb_g)          // broadcast over tokens
t = ggml_mul_mat(lb_v, t)      // [r_pad, n], or [r, n] in version 2
t = ggml_pad(t, r_pad - r)     // only when lb_v is unpadded
y = ggml_mul_mat(lb_u, t)      // [out, n]
(+ the same for the residual path, + bias)
```

This is exact up to two things: scales converted from bf16 to fp16 (the converter checks every value is in fp16 range and reports the largest relative change), and llama.cpp's usual quantization of activations inside the `Q1_0` dot product (typically to `Q8_0`). The tests measure both.

**Padding.** `r_pad` is `r` rounded up to a multiple of 128. In `lb_v`, padded rows get `d = 0`, so their outputs are exactly 0. In `lb_u`, padded columns multiply those zeros (write 0). With format version 2 (`--unpadded-v`), `lb_v` keeps the true rank `r` and the loader pads the latent with zeros to `r_pad` before `lb_u`; see [FORMAT.md](FORMAT.md#gguf-layout). The input width must already be a multiple of 128. SmolLM2-135M has width 576 and fails this check.

**Optional fusion.** The second stage of the two paths can be merged: `y = [U' | U'_R] · [V'(g ⊙ x) ; V'_R(g_R ⊙ x)]`. It saves one matmul per layer; the first stage cannot be merged because `g` and `g_R` differ. Only worth doing if the benchmarks show a gain.

## llama.cpp changes

Kept small and simple enough to propose upstream later. They follow llama.cpp's CONTRIBUTING.md and its code style: reuse existing infrastructure, short comments in plain English, ASCII only in code and comments.

1. **Loading**: for supported architectures, when `<base>.lb_v` exists, load the LittleBit tensors for that projection instead of `<base>.weight`. Validate shapes: `r >= 1` and `r_pad` is `r` rounded up to a multiple of 128. Refuse expert tensors (`ffn_*_exps.lb_*`).
2. **Graph**: every weight matmul in llama.cpp already goes through `llm_graph_context::build_lora_mm`. Route projections that have LittleBit tensors through one helper there, so the per-architecture model code barely changes.
3. **Metadata**: GGUF keys `littlebit.format_version` (1 or 2), `littlebit.residual`, `littlebit.eff_bits` (informational), `littlebit.use_itq` (informational).
4. LoRA adapters on LittleBit projections: reject with a clear message in the first version.

First architectures: Llama (including SmolLM-style models), Qwen2, Qwen3.

llama.cpp is included as a git submodule pinned to a commit that has `Q1_0`; the changes live as plain diffs in `patches/` (no commit messages) applied by `scripts/build.sh`. Nothing is committed inside the submodule. The pinned commit is bumped on purpose, never silently.

## Converter

`convert_littlebit_to_gguf.py`:

- Reuses llama.cpp's own Python tooling (`gguf-py`, and the model classes in `convert_hf_to_gguf.py` for Llama/Qwen2/Qwen3) for the tokenizer, hyperparameters, norms and embeddings, and only takes over the compressed projections.
- Reads the packed int32 arrays directly; unpacks to floats only for validation.
- `--embd-type` and `--output-type` (`f16`, `q8_0`, `q6_k`, `q4_k`) for token embeddings and the output layer. The same types as pinned `llama-quantize` are used for incompatible rows: Q4_K falls back to Q5_0, Q6_K to Q8_0, and an incompatible 32-column block falls back to F16. Tied weights share one tensor when the effective storage types match.
- `--unpadded-v`: store `lb_v` at the true rank and write format version 2 (version 1 when every rank is a multiple of 128).
- `--validate N`: for N random layers, compares the GGUF tensors' result (dequantized in numpy) with the reference float computation and prints the error.
- Prints a summary: per-layer rank, padding, bits per weight, total file size and how much of it is embeddings.

## Testing

- **Synthetic checkpoints**: a script that turns a real small Hugging Face model (Qwen2.5-0.5B and Qwen3-0.6B, with an aligned tiny fixture for Llama) into a format-correct LittleBit checkpoint using the paper's initialisation (truncated SVD, signs of the factors, rank-1 fits of their magnitudes for the scales, optional residual), with no training. Quality is meaningless; it exists for equivalence tests.
- **Layer level**: converter output versus the reference float computation, exact up to fp16 scale rounding.
- **Model level**: logits from the patched llama.cpp versus a plain PyTorch implementation of the same synthetic weights, on fixed prompts: top-1 agreement, max absolute difference and KL divergence, with tolerances checked against activation rounding and cache precision measurements.
- **Smoke**: `llama-cli` generates text from a converted file; `llama-bench` reports speed. File sizes are recorded.

## Milestones

1. Converter, llama.cpp patch for Llama/Qwen2/Qwen3 on CPU, tests, sizes and CPU speed.
2. Check the other backends that already support `Q1_0` (Metal, CUDA, Vulkan) where hardware is available; more architectures.
3. Upstream: first open an issue on llama.cpp to discuss the idea and check for related work, then propose the change if maintainers are interested. After that, Ollama and LM Studio can load these files.
