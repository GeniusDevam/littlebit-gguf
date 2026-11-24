# LittleBit checkpoint format

Notes on the files written by the official LittleBit training code ([SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)), from studying how its checkpoints are saved and loaded. They describe the format only. No code from that repository is part of this project. The same notes live in the sister project littlebit.cpp.

The GGUF layout this project writes is described in [GGUF layout](#gguf-layout) at the end.

LittleBit and LittleBit-2 write the **same** format. LittleBit-2 changes only how training is initialised (`use_itq: true`), so a runtime does not need to tell them apart.

## Files in a checkpoint directory

| File | Contents |
|---|---|
| `config.json` | The base Hugging Face model config, plus the quantization fields below and `use_cache: true` |
| `littlebit_config.json` | The quantization fields on their own |
| `model.safetensors` | All tensors. Large models may be sharded, with `model.safetensors.index.json`. Older checkpoints may use `model_state.pt` |
| tokenizer files | As saved by Hugging Face |

Quantization fields: `quant_func`, `eff_bit`, `split_dim`, `residual`, `kv_factor`, `min_split_dim`, `quant_mod`, `use_itq`, `itq_n_iter`. Older checkpoints without `littlebit_config.json` need these supplied by the user.

## Which layers are compressed

Every `nn.Linear` in the model except `lm_head`. Token embeddings, norms and `lm_head` are stored normally (16-bit). In Llama/Qwen-style models that means `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in every block. `kv_factor` only changes the rank chosen for `k_proj`/`v_proj` at training time; at load time read ranks from the shape tensors. Phi-3 models are a special case (the fused QKV projection is split first) and are out of scope for now.

## Tensors for one compressed layer

For a layer with prefix `P` (for example `model.layers.0.mlp.up_proj`), `in` input features, `out` output features and rank `r`:

| Name | dtype | Shape | Meaning |
|---|---|---|---|
| `P.U_packed` | int32 | `[out, ceil(r/32)]` | sign matrix `U±`, `out × r` |
| `P.U_shape` | int64 | `[2]` | `(out, r)` |
| `P.V_packed` | int32 | `[r, ceil(in/32)]` | sign matrix `V±`, `r × in` |
| `P.V_shape` | int64 | `[2]` | `(r, in)` |
| `P.u1` | bf16 | `[1, out]` | output scale `h` |
| `P.u2` | bf16 | `[1, r]` | half of the rank scale |
| `P.v1` | bf16 | `[1, r]` | other half of the rank scale |
| `P.v2` | bf16 | `[1, in]` | input scale `g` |
| `P.bias` | as saved | `[out]` | only if the original layer had a bias (for example Qwen2 q/k/v) |

The rank scale is `l = v1 ⊙ u2` (element-wise). Scales can be negative.

If `residual` is true, a second path is stored with the same names plus `_R`: `U_R_packed, U_R_shape, V_R_packed, V_R_shape, u1_R, u2_R, v1_R, v2_R`. Its rank is the same as the primary path's.

Informational buffers may also appear: `_eff_bit_target`, `_split_dim_final`, `_eff_bit_actual` (cast to bf16 on save). They are not needed for inference.

## Bit packing

- Matrices are packed row by row into 32-bit words, **least significant bit first**: column `c` of row `i` is bit `c % 32` of word `c // 32` in row `i`.
- **Bit 0 means +1, bit 1 means −1.**
- Each row is padded to a multiple of 32 columns with +1 (bit 0).
- Words are stored as signed int32; read them as uint32. On a little-endian machine, byte `k` of a word holds columns `32w + 8k` to `32w + 8k + 7`, least significant bit first.

## Forward pass

The reference computes each path in this order (`x` has shape `[tokens, in]`):

```
t = x ⊙ v2            # input scale g
t = t @ V±ᵀ           # [tokens, r]
t = t ⊙ (v1 ⊙ u2)     # rank scale l
t = t @ U±ᵀ           # [tokens, out]
y = t ⊙ u1            # output scale h
```

With `residual`, the `_R` path is computed the same way and added. The bias, if any, is added last.

## Bits per weight

How the reference counts a layer's size (`a = in`, `b = out`, `r` = rank, scales counted at 16 bits):

- with residual: `(2·r·(a + b + 16) + 32·(a + b)) / (a·b)`
- without: `(r·(a + b + 16) + 16·(a + b)) / (a·b)`

Embeddings and `lm_head` are not included in that figure, and in small models they are most of the file.

# GGUF layout

How a LittleBit model is stored in a GGUF file for llama.cpp. Format versions 1 and 2 exist. The loader in this repository reads both. The converter writes version 1 by default and version 2 with `--unpadded-v`. Two other implementations, [luben93/littlebit-llamacpp](https://github.com/luben93/littlebit-llamacpp) and [ScarionWorkspace/littlebit-gguf](https://github.com/ScarionWorkspace/littlebit-gguf), agreed these parts in [issue 1](https://github.com/GeniusDevam/littlebit-gguf/issues/1): the tensor names, the four metadata keys, the unpadded `lb_v`, the padded `lb_u`, and format versions 1 and 2. Two parts of this section are proposals of this project that have not been posted there yet: the `littlebit.x.*` key namespace and the mixture-of-experts layout. They are marked below.

Shapes are given in ggml order: `ne[0]` is the contiguous dimension, the length of one row.

## Metadata

| Key | Type | Meaning |
|---|---|---|
| `littlebit.format_version` | uint32 | `1` or `2`. A loader refuses any other value with an error that names the value. |
| `littlebit.residual` | bool | The file has a residual path for every projection. |
| `littlebit.eff_bits` | float32 | Target bits per weight. Must be finite and above 0. |
| `littlebit.use_itq` | bool | Informational. LittleBit-2 initialisation was used. |
| `littlebit.x.*` | any | Implementation-specific. Loaders ignore every key in this namespace. Proposal, not yet agreed. |

All other keys are the ones the stock llama.cpp converter writes for the architecture. Supported architectures are `llama`, `qwen2` and `qwen3`.

## Tensors for one projection

For the projection `blk.N.<proj>` with `width` input features, `out` output features and rank `r`, let `r_pad` be `r` rounded up to a multiple of 128. `<proj>` is one of `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`. The file has no `blk.N.<proj>.weight` tensor for it.

| Tensor | Type | Shape | Contents |
|---|---|---|---|
| `blk.N.<proj>.lb_g` | F32 | `[width]` | Input scale `g`. |
| `blk.N.<proj>.lb_v` | Q1_0 | `[width, r_pad]` in version 1, `[width, r]` or `[width, r_pad]` in version 2 | Rows of `V±`. Row `k` has every block scale `d` set to the rank scale `l_k`. |
| `blk.N.<proj>.lb_u` | Q1_0 | `[r_pad, out]` | Rows of `U±`. Row `i` has every block scale `d` set to the output scale `h_i`. |

With `littlebit.residual`, the tensors `lb_g_r`, `lb_v_r` and `lb_u_r` hold the second path. It has the same `width`, `r` and `r_pad` as the first.

`lb_u` is always padded because `r` is its row length, and a Q1_0 row is a whole number of 128-bit blocks. `lb_v` has `r` rows, so it can hold the true rank. `width` must be a multiple of 128.

Q1_0 follows llama.cpp: a block holds an fp16 `d` and 128 sign bits, bit `j` of the block is bit `j % 8` of byte `j / 8`, and a set bit means `+d`. This is the opposite of the checkpoint, where bit 0 means +1. Scales are stored as fp16. A scale that rounds to zero is stored as zero.

Padding is inert. The extra rows of a padded `lb_v` have `d = 0`. The extra columns of `lb_u` are multiplied by latent values that are zero, and a writer should clear their sign bits.

## Forward pass

For each path, with `x` the input of the projection:

```
latent = V · (g ⊙ x)          # r values
latent = zero-pad to r_pad    # only when lb_v has r < r_pad rows
y      = U · latent
```

With `littlebit.residual`, the output is the sum of both paths. Any bias is added afterwards, as for a dense projection.

## Version rules

- **Version 1:** every `lb_v` has `r_pad` rows.
- **Version 2:** an `lb_v` may have the true rank `r` as its row count. Padded files are still valid.
- **Writers:** write version 2 only when some `lb_v` is unpadded. When every rank is already a multiple of 128 the two layouts are the same bytes, so write version 1.
- **Readers:** take `r` from the shape of `lb_v` and `r_pad` from the shape of `lb_u`. Require `r >= 1`, `r_pad` equal to `r` rounded up to a multiple of 128, and `width` a multiple of 128. Zero-pad the latent when `r < r_pad`. Reject anything else with an error that names the tensor. Accept versions 1 and 2 and refuse all others before reading tensors. A reader may accept an unpadded `lb_v` in a version 1 file, and this repository does. Writers must not produce one.

## Mixture of experts

This part is a proposal of this project, not yet agreed with the other implementations, defined so that they can agree on it. This repository does not implement it. It does not load or run expert tensors yet.

The stacked expert tensors `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` keep llama.cpp's own `[n_embd, n_ff, n_expert]` layout, with the expert index last. For each of them, with `n_expert` experts:

| Tensor | Type | Shape |
|---|---|---|
| `blk.N.<exps>.lb_g` | F32 | `[width, n_expert]` |
| `blk.N.<exps>.lb_v` | Q1_0 | `[width, r, n_expert]` |
| `blk.N.<exps>.lb_u` | Q1_0 | `[r_pad, out, n_expert]` |

With `littlebit.residual`, `lb_g_r`, `lb_v_r` and `lb_u_r` hold the second path in the same shapes.

- **One rank per tensor:** `r` is the same for every expert of a stacked tensor, which stacking needs. The unpadded `lb_v` rule applies per expert, so `lb_v` may have `r < r_pad`.
- **Gate and up stay separate:** LittleBit does not use a merged `ffn_gate_up_exps` tensor, because the two projections have different factors.
- **Router:** `ffn_gate_inp` is an ordinary tensor at normal precision.
- **Shared experts:** `ffn_gate_shexp`, `ffn_up_shexp` and `ffn_down_shexp` are ordinary two-dimensional LittleBit projections, as above.
- **Loaders without expert support:** a loader that has not implemented expert tensors must refuse a file that contains them, with a clear error, and must not guess a layout. The loader in this repository does this in two steps. Its architecture check accepts only `llama`, `qwen2` and `qwen3`. That is not enough, because a `llama` file can have experts, so the loader also throws `LittleBit expert tensors are not supported: <tensor>` when any `ffn_*_exps.lb_*` tensor exists.
