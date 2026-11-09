# Third-party notices

eGemma's original code is licensed under the [MIT License](LICENSE).
The third-party material below retains its own license. No model weights or
training datasets are bundled in this repository.

---

### Adapted code

[`src/models/corpus_gpt.py`](src/models/corpus_gpt.py) adapts the GPT model of
[nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (its `model.py`
and `train.py`, as of commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`). It
keeps nanoGPT's architecture: pre-norm blocks of causal self-attention and a GELU
MLP, learned position embeddings, and the token embedding tied to the output
head. From the training recipe it keeps weights drawn from N(0, 0.02), AdamW with
betas (0.9, 0.95) and weight decay 0.1, and gradient clipping at 1.0.

eGemma's version is written compactly, is much smaller (4 layers, 4 heads and 64
dimensions by default), and trains on one caller's corpus for one request,
sentence by sentence with begin and end tokens and padding that the loss
ignores. It also departs from nanoGPT's recipe:

- weight decay applies to every parameter, not only to matrices and embeddings;
- biases keep PyTorch's default initialization rather than zeros, and the
  residual projections are not scaled down by depth;
- dropout is 0.1;
- the learning rate, peaking at 3·10⁻³, is a 100-step warmup multiplied by a
  cosine over all steps that decays to zero, rather than a warmup followed by a
  cosine decay to a floor.

nanoGPT is covered by the MIT License, copyright (c) 2022 Andrej Karpathy,
reproduced in [`LICENSES/nanoGPT-MIT.txt`](LICENSES/nanoGPT-MIT.txt).

---

### Semantic parsing dependencies

| Package | License | Source |
| --- | --- | --- |
| PENMAN 1.3.1 | MIT; copyright 2016–2019 Michael Wayne Goodman | [Upstream](https://github.com/goodmami/penman/tree/v1.3.1), [license copy](LICENSES/PENMAN-MIT.txt) |
| Transformers 5.10.1 | Apache-2.0 | [Upstream license](https://github.com/huggingface/transformers/blob/v5.10.1/LICENSE), [license copy](LICENSES/Apache-2.0.txt) |

These are installed dependencies, not vendored or relicensed source. PENMAN's
AMR model is used directly; no SEMCAT code or caller-specific PENMAN patches
are included. Other Python and JavaScript dependencies retain the licenses and
notices distributed with their packages. This document is not a complete
inventory of transitive dependencies; preserve their notices when distributing
an environment or container.

---

### Downloaded models

| Configured model | Published license or terms |
| --- | --- |
| `google/embeddinggemma-300m` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms), as linked by its [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) |
| `xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2` | The [pinned checkpoint card](https://huggingface.co/xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2/blob/12f7e78e3dfa6f37ddbbf77c4c152b4fc39bc512/README.md) declares MIT. The separate [AMRBART code repository](https://github.com/goodbai-nlp/AMRBART/blob/main/LICENSE) is also MIT, copyright 2022 xfbai. |
| `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` | Apache-2.0, as declared by the [pinned checkpoint card](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/blob/d1c082be9cf3c8a514acf63b8761f4b41935842e/README.md); see the [Gemma 4 model card](https://huggingface.co/google/gemma-4-26B-A4B-it) |

eGemma's MIT license does not replace these model terms or grant rights to their
training datasets. If you redistribute weights, include the applicable terms
and notices.
