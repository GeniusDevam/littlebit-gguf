# littlebit-gguf

Run **sub-1-bit LittleBit and LittleBit-2 language models in llama.cpp**, using the 1-bit `Q1_0` tensor type llama.cpp already ships instead of writing new kernels.

> **This is a research project.** It is independent and experimental, and it is not affiliated with or endorsed by Samsung or the llama.cpp project. Expect breaking changes and rough edges.

## The idea

LittleBit (Samsung Research, NeurIPS 2025) and LittleBit-2 (ICML 2026) compress every linear layer of a transformer into two small sign matrices and three scale vectors, reaching 0.1 to 1 bit per weight. A layer computes:

```
y = h ⊙ U±( l ⊙ V±( g ⊙ x ) )
```

llama.cpp's `Q1_0` type stores weights in blocks of 128 bits with one fp16 scale `d` per block, and each bit means `+d` or `−d`. Because that scale belongs to a block within a single row, LittleBit's row scales fold straight into it:

- `V' = diag(l) · V±` : every block in row `k` gets `d = l_k`
- `U' = diag(h) · U±` : every block in row `i` gets `d = h_i`

So a LittleBit layer becomes one element-wise multiply and two ordinary `Q1_0` matrix multiplies:

```
y = U' · ( V' · ( g ⊙ x ) )          (and the same again for the residual path)
```

No new kernels. Anything that already runs `Q1_0` runs these layers too.

## What it costs

`Q1_0` needs each row length to be a multiple of 128, so the rank is padded up to the next multiple of 128 (padded rows get `d = 0` and add nothing), and each block carries 16 bits of scale. Estimated bits per weight for the compressed layers:

| Model | Trained at | In GGUF |
|---|---|---|
| Llama-2-7B | 1.0 | ~1.15 |
| Llama-2-7B | 0.55 | ~0.69 |
| Llama-2-7B | 0.1 | ~0.12 |
| Qwen2.5-0.5B | 0.55 | ~0.80 |
| Qwen2.5-0.5B | 0.1 | ~0.47 |

Large models lose little. Small models at very low bit rates lose a lot to padding, because their ranks are tiny. Training with ranks that are multiples of 128 avoids it, and the converter warns when padding is heavy. (Estimates from the shape arithmetic; the converter prints the real figure for each file.)

The other half of the story is the layers LittleBit leaves alone. The reference keeps token embeddings and the output layer in 16 bits, and in small models they are most of the file: about 272 MB for Qwen2.5-0.5B next to about 4 MB of LittleBit layers at 0.1 bits per weight. The converter can store them as `Q8_0`, `Q6_K` or `Q4_K` instead.

## Status

| Component | Status |
|---|---|
| Design and format notes | done, see [docs/DESIGN.md](docs/DESIGN.md) and [docs/FORMAT.md](docs/FORMAT.md) |
| Converter: LittleBit checkpoint to GGUF | planned, first milestone |
| llama.cpp patch: Llama, Qwen2, Qwen3 | planned, first milestone |
| Correctness tests and CPU benchmarks | planned, first milestone |
| More architectures (Gemma, Phi) | planned |
| Upstream proposal to llama.cpp | planned |

Until the changes are upstream, LittleBit GGUF files need the patched build from this repo. Stock llama.cpp, Ollama and LM Studio will not load them yet.

## Related work

- LittleBit: [arXiv:2506.13771](https://arxiv.org/abs/2506.13771). LittleBit-2: [arXiv:2603.00042](https://arxiv.org/abs/2603.00042).
- Official training code: [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) (CC BY-NC 4.0). No code from it is used or redistributed here.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), whose `Q1_0` type was added for PrismML's 1-bit Bonsai models.
- Sister project: [littlebit.cpp](https://github.com/GeniusDevam/littlebit.cpp), dedicated CPU kernels for LittleBit layers.

## License

MIT, see [LICENSE](LICENSE). llama.cpp is MIT licensed. Model weights keep their own licenses.

## Citation

Please cite the original papers:

```bibtex
@inproceedings{lee2025littlebit,
  title     = {LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author    = {Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}

@inproceedings{lee2026littlebit2,
  title     = {LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author    = {Lee, Banseok and Kim, Youngmin},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
