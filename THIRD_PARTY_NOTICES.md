# Third-party notices

eGemma's original code is licensed under the [MIT License](LICENSE).
The third-party material below retains its own license. No model weights or
training datasets are bundled in this repository.

## Bundled reference template

[`docs/template.txt`](docs/template.txt) is OpenAI's GPT-OSS chat template from
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/chat_template.jinja),
revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
It is covered by the upstream [Apache-2.0 license](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/LICENSE),
reproduced in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

The template body is unchanged; eGemma adds an attribution comment. Before that
comment, its SHA-256 is
`a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146`.
No separate upstream NOTICE file is present at this revision.
The Python project metadata records `MIT AND Apache-2.0` to account for this
bundled material; eGemma's own code remains MIT.

## Adapted code

[`src/models/corpus_gpt.py`](src/models/corpus_gpt.py) adapts the GPT model and
training recipe of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
Karpathy (its `model.py` and `train.py`, as of commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`): pre-norm blocks of causal
self-attention and a GELU MLP, the token embedding tied to the output head, the
weights drawn from N(0, 0.02), AdamW with betas (0.9, 0.95) and weight decay 0.1,
and a warmed-up cosine learning rate. eGemma's version is written compactly,
smaller and trained on a caller's corpus for one request.

nanoGPT is covered by the MIT License, copyright (c) 2022 Andrej Karpathy,
reproduced in [`LICENSES/nanoGPT-MIT.txt`](LICENSES/nanoGPT-MIT.txt).

## Semantic parsing dependencies

| Package | License | Source |
| --- | --- | --- |
| PENMAN 1.3.1 | MIT; copyright 2016–2019 Michael Wayne Goodman | [Upstream](https://github.com/goodmami/penman/tree/v1.3.1), [license copy](LICENSES/PENMAN-MIT.txt) |
| Transformers 4.57.6 | Apache-2.0 | [Upstream license](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE), [license copy](LICENSES/Apache-2.0.txt) |

These are installed dependencies, not vendored or relicensed source. PENMAN's
AMR model is used directly; no SEMCAT code or Tendora-specific PENMAN patches
are included. Other Python and JavaScript dependencies retain the licenses and
notices distributed with their packages. This document is not a complete
inventory of transitive dependencies; preserve their notices when distributing
an environment or container.

## Downloaded models

| Configured model | Published license or terms |
| --- | --- |
| `google/embeddinggemma-300m` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms), as linked by its [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) |
| `xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2` | The [pinned checkpoint card](https://huggingface.co/xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2/blob/12f7e78e3dfa6f37ddbbf77c4c152b4fc39bc512/README.md) declares MIT. The separate [AMRBART code repository](https://github.com/goodbai-nlp/AMRBART/blob/main/LICENSE) is also MIT, copyright 2022 xfbai. |
| GPT-OSS | [Apache-2.0](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/LICENSE); see the upstream [usage policy](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/USAGE_POLICY) and the terms supplied with your chosen GGUF conversion |
| `google/gemma-3-12b-it-qat-q4_0-gguf` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms), as linked by its [model card](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf) |

eGemma's MIT license does not replace these model terms or grant rights to their
training datasets. If you redistribute weights, include the applicable terms
and notices. Gemini is an optional hosted service governed by its provider's
[API terms](https://ai.google.dev/gemini-api/terms).
