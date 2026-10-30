# eGemma API Reference

- **Base URL:** `http://localhost:8000`
- **Authentication:** `Authorization: Bearer <WORKBENCH_API_KEY>` on every
  endpoint except `/rate-limits`, `/docs` and `/`. `/health` answers without the
  key but gives its model detail only with it.
- **Request size:** a body is limited before it is parsed or authenticated: an
  upload to 5 MB plus 1 MiB of multipart framing, any other body to 16 MiB
  (`MAX_REQUEST_BYTES`). Past that, HTTP 413.

| Area | Endpoints | Model |
| --- | --- | --- |
| [Embeddings](#embeddings) | `/embed/texts`, `/embed/tokens`, `/embed` | EmbeddingGemma 300M |
| [Corpus transformer](#corpus-transformer) | `/corpus/gpt` | nanoGPT-design network trained per request |
| [AMR](#abstract-meaning-representation) | `/amr/parse`, `/amr/parse-batch` | AMRBART |
| [Summarization](#summarization) | `/summarize` | Gemma 4 |
| [AST parsing](#ast-parsing) | `/parse-ast` | none (Python `ast`) |
| [Monitoring](#monitoring) | `/health`, `/rate-limits` | |

---

## Embeddings

All three endpoints use the pinned local EmbeddingGemma revision, which is
always loaded.

### POST /embed/texts

Embed 1–256 texts in one request. The request selects one supported prompt and
one Matryoshka dimension (128 by default; 256, 512 or 768 explicitly); the
response repeats the exact model, revision, dimension and prompt.

```bash
curl -X POST "http://localhost:8000/embed/texts" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts":["candidate relation","canonical relation"],"dimensions":768,"prompt_name":"STS"}'
```

JSON is wire transport only; callers decide durable representation. Each input
must contain 1–16,384 characters by default. The encoder has a 2,048-token
window, including any task prompt, and may truncate beyond it; split longer
inputs before embedding when full source coverage is required.

### POST /embed/tokens

A word read in its sentence. For each character span asked, the response gives
the mean of EmbeddingGemma's last-layer states over the sub-word tokens the
span overlaps. Use it when the same word should be compared as it is used in
different texts, which a whole-text embedding cannot separate.

```bash
curl -X POST "http://localhost:8000/embed/tokens" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"items":[{"text":"The valve closes the tank.","spans":[[4,9],[21,25]]}]}'
```

```json
{
  "model": "google/embeddinggemma-300m",
  "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
  "layer": "last",
  "dimensions": 768,
  "encoding": "json",
  "items": [{"vectors": [[0.012, -0.341, ...], [0.207, 0.055, ...]]}]
}
```

| Field | Meaning |
| --- | --- |
| `items[].text` | The text, read as written: no task prompt is added |
| `items[].spans` | 1–64 `[start, end)` character offsets (Python string indices, i.e. code points); each must satisfy `0 <= start < end <= len(text)` |
| `encoding` | `"json"` (default): vectors as JSON numbers. `"base64"`: each vector as its little-endian float32 bytes, base64-encoded (4,096 characters for 768 floats, about a quarter of the JSON) |
| `items[].vectors` | One vector per span, in order; `null` where the span overlaps no token. Special tokens never count. |

Vectors are the transformer's full-width output before EmbeddingGemma's pooling,
projection and normalization, so they are not Matryoshka embeddings and are
not comparable with `/embed/texts` vectors. Each state reads the whole text,
before and after the span. A request holds 1–256 items, each text 1–16,384
characters, and asks for at most 8,192 vectors in all (`EMBED_TOKENS_MAX_VECTORS`).
Out-of-range spans, empty texts and too many vectors return HTTP 422, and so does
a text longer than the model's 2,048-token window, with its token count; texts
are never cut.

To decode a base64 vector: `numpy.frombuffer(base64.b64decode(v), "<f4")` in
Python. In Node.js, copy the bytes out of Buffer's shared pool first:
`const b = Buffer.from(v, "base64"); new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length))`
(on a little-endian machine, as nearly all are).

### POST /embed

Embed one uploaded file.

| Query parameter | Default | Description |
| --- | --- | --- |
| `dimensions` | 128 | One or more of 128, 256, 512 and 768 (repeat the parameter); one `embedding_<n>d` field each |
| `prompt_name` | document | A supported EmbeddingGemma prompt: query, document, STS, … |
| `title` | null | Title of a document; it replaces "none" in the `document` prompts' `title: none \| text:` |

```bash
curl -X POST "http://localhost:8000/embed?dimensions=256&prompt_name=document" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.txt"
```

```json
{
  "model": "google/embeddinggemma-300m",
  "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
  "embedding_256d": [0.123, -0.456, ...]
}
```

A file longer than the model's 2,048-token window, prompt included, returns
HTTP 422 with its token count: it is never cut. Uploads past 5 MB return
HTTP 413; binary or non-UTF-8 files return HTTP 400.

---

## Corpus transformer

### POST /corpus/gpt

Train a small transformer from scratch on the caller's sentences, then read it
back. The network follows [nanoGPT](https://github.com/karpathy/nanoGPT)'s
design (pre-norm causal self-attention, a GELU MLP, tied token embedding); see
[third-party notices](../THIRD_PARTY_NOTICES.md).

**Why.** A pretrained model brings what it learned elsewhere. This network knows
nothing but the sentences it is given, so its vectors show what *the corpus
alone* says about its own vocabulary: which words it uses in the same places and
contexts. Set them beside a pretrained model's view of the same words, the bare
word from `/embed/texts` or the word in its sentences from `/embed/tokens`:

- where both agree, the corpus supports the general reading of a word;
- where they differ, either the pretrained model contributes knowledge the
  corpus does not contain, or the corpus uses the word its own way;
- training twice, on a corpus and on that corpus plus new text (for example a
  set of questions), shows what the new text changes.

**Request.** The caller splits sentences into words and names what to read
back, under keys of its choosing:

| Field | Default | Bounds | Meaning |
| --- | --- | --- | --- |
| `sentences` | required | 1–20,000 sentences of at most 256 words; at most 20,000 distinct words | Each sentence as a list of words; the vocabulary is exactly these words |
| `words` | `{}` | words of at most 256 characters | Key → word: return that word's token-embedding row (the learned weights). A key whose word is absent is omitted. |
| `positions` | `{}` | at most 200,000 places in all | Key → `[[sentence, word], …]`, zero-based: return the hidden state at each layer at those places, averaged. An out-of-range place returns HTTP 422. |
| `steps` | 1500 | 1–20,000 | Training steps (batches of up to 32 sentences) |
| `seed` | 7 | | Seeds initialization and batch order |
| `dimensions` | 64 | 8–512 | Width; must divide evenly among `heads` |
| `layers` | 4 | 1–12 | Transformer blocks |
| `heads` | 4 | 1–16 | Attention heads |
| `threads` | 2 | 1–8 | CPU threads for this training |
| `stream` | true | | Server-sent progress events; `false` returns one JSON document |

What is read back is bounded too: `dimensions × (words + positions × layers)`
floats, at most 2,000,000 (about 40 MB of JSON; some 6,000 words and positions
at the defaults). Past any bound the request is refused with HTTP 422 before it
trains.

```bash
curl -N -X POST "http://localhost:8000/corpus/gpt" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sentences": [["the","pump","feeds","the","tank"],
                  ["the","valve","closes","the","tank"]],
    "words": {"tank": "tank", "pump": "pump"},
    "positions": {"tank": [[0,4],[1,4]], "pump": [[0,1]]},
    "steps": 300
  }'
```

**Streamed events** (`text/event-stream`):

```text
event: progress
data: {"step": 25, "steps": 300, "loss": 2.1043}

event: result
data: {"model": {...}, "wte": {...}, "layers": [...]}
```

`progress` arrives every 25 steps and at the end; the stream ends with one
`result` or one `error` (`{"detail": "..."}`). A client that disconnects, streamed
or not, stops its training at the next step.

One network trains at a time and two more requests may wait their turn; a request
beyond that gets HTTP 503 with `Retry-After`.

**Result** (the `result` event, or the whole body with `"stream": false`):

```json
{
  "model": {
    "architecture": "nanoGPT, 4 layers, 4 heads, 64 dimensions, tied embedding",
    "parameters": 201088,
    "vocab": 9,
    "tokens": 10,
    "steps": 300,
    "seed": 7,
    "trainLoss": 0.1092
  },
  "wte": {"tank": [64 floats], "pump": [64 floats]},
  "layers": [{"tank": [64 floats], "pump": [64 floats]}, ...]
}
```

`layers` has one object per block, first to last. The network is causal: a state
at a word has seen that word and the ones before it in its sentence, not the ones
after, unlike `/embed/tokens`, which reads the whole text. Vectors are raw, in this
network's own space: compare them within one result, not across trainings or
with EmbeddingGemma vectors directly. To set two sources side by side, compare
their geometry instead, for example each word's nearest neighbours or the rank
correlation of all pairwise similarities. States at one layer tend to share a
common direction, so centre them (subtract the mean over keys) before comparing
by cosine.

**Reading it.** On small corpora the parts of the network tend to say different
things. `wte` mostly reflects which words are *said together*. Deeper layers'
states move toward which words *share roles*, the places one word could stand
in for another. Fitting a small corpus is the point, but the role structure
tends to appear early and then give way to memorized co-occurrence. So treat
`steps` as a setting to compare, not a bigger-is-better budget, and repeat over
a few seeds before trusting a difference. Corpus size is the real limit: a few
thousand tokens give a weak signal.

Training runs on CPU, one network at a time; a concurrent request waits for the
running one. A few thousand words take about half a minute on two threads.
Nothing is written to disk: the corpus and the network live for one request.

---

## Abstract Meaning Representation

Loaded when `AMRBART_MODEL_ENABLED=true` (the default); otherwise both
endpoints return HTTP 503.

### POST /amr/parse

Parse natural-language text into an attributed AMRBART candidate graph.

```bash
curl -X POST "http://localhost:8000/amr/parse" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The parser passes the tests, but changes the dependency graph."
  }'
```

The response contains:

| Field | Meaning |
|-------|---------|
| `model`, `revision` | Exact parser identity |
| `device` | Execution device used for the query |
| `input_tokens`, `output_tokens` | Complete source-token count including boundaries, and generated-token count; output is zero when inference is skipped |
| `linearized` | Indented pointer graph for valid parses; raw model output if decoding fails |
| `linearized_raw` | Exact AMRBART pointer-token return, retained without formatting |
| `penman` | Validated standard PENMAN graph, or `null` |
| `top`, `triples` | Root and ordered typed triples from that validated graph |
| `fragments` | Diagnostic relations from an invalid graph; never a substitute for validated triples |
| `decode_status`, `decode_error` | Mechanical decoding result |

The endpoint does not claim that the graph is factually true, source-aligned,
or authorized. It exposes one learned semantic parse for downstream inspection.
JSON escapes newlines inside both formatted graph strings. Use
`jq -r '.linearized'` or `jq -r '.penman'` to print their indentation.

Decoding uses PENMAN's AMR model. Lexical roles such as `:consist-of` and
`:prep-out-of` keep their direction; genuine inverses such as `:ARG0-of` are
decoded as reversed `:ARG0` edges. The full output must contain one complete
graph with concepts and relation targets. eGemma does not repair missing
structure or apply downstream semantic canonicalization.

### POST /amr/parse-batch

Parse a bounded batch sequentially, returning the same result schema for each
input in its original order:

```bash
curl -X POST "http://localhost:8000/amr/parse-batch" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts":["The engineer reported a failure.","The repair needs approval."]}'
```

Both AMR endpoints return HTTP 200 for these per-input results:

| `decode_status` | Meaning |
| --- | --- |
| `ok` | A complete graph was decoded; `penman`, `top`, and `triples` are available. |
| `invalid` | Inference ran but decoding failed. Raw output, the error, and diagnostic fragments are retained; `penman` and `top` are `null`, and `triples` is empty. |
| `input_too_long` | The complete source exceeds the token budget. Inference is skipped; graph strings are empty, `penman` and `top` are `null`, and triples/fragments and output tokens are empty/zero. `decode_error` states the count and limit. |
| `timed_out` | Generation reached its deadline (`AMRBART_MAX_SECONDS`, 45 s by default). What the search held then is not the graph it would have returned, so none is kept: the fields are empty as for `input_too_long`, and `decode_error` states the deadline. |

The default limits are 1–64 inputs per batch, 1–16,384 characters per input,
400 source tokens, and 1,024 generated tokens. Source tokens include the text
boundaries and the three appended graph tokens:
`<s> text </s> <AMR> <mask> </AMR>`. The source is never silently truncated;
split long inputs explicitly. The character bound is a separate request limit,
not a guarantee that the text fits the tokenizer budget.

An invalid batch shape or character bound returns HTTP 422 for the request.
Invalid graphs and token-overflow results do not discard other batch items;
an unexpected inference/runtime failure returns HTTP 500. A batch whose client
disconnects stops before its next text and answers 499; the texts left are not
parsed.

A graph counts as finished when the output ends with `</AMR>`; only output that
has not ended by the deadline is `timed_out`.

---

## Summarization

### POST /summarize

Summarize a code, Markdown, or log file with the local Gemma 4 model and a
[persona](../personas/README.md). Loaded when `SUMMARY_MODEL_ENABLED=true` (the
default); otherwise returns HTTP 503.

The language comes from the file name: `.md`, `.markdown`, `.mkd` and `.mdx` are
Markdown and use the `docs/` personas; `.log` files, rotated ones such as
`app.log.1` included, are logs and are condensed first (repeated lines collapsed
where they first occur, in order); other extensions name a language or default to
"code".

| Form parameter | Default | Description |
| --- | --- | --- |
| `file` | required | File to summarize: UTF-8 text, at most 5 MB |
| `persona` | `developer`, or `assistant` for Markdown | Persona file name, e.g. `log_analyst`, `security_validator` |
| `max_tokens` | 512 | Maximum summary tokens, at least 1 |
| `temperature` | 0.7 | Sampling temperature, 0 to 2 |

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "persona=developer"
```

```json
{
  "language": "Python",
  "summary": "This module implements...",
  "truncated": false,
  "persona": "code/developer",
  "model": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
  "revision": "d1c082be9cf3c8a514acf63b8761f4b41935842e"
}
```

`persona` is the persona file used: a name with no file falls back to its folder's
`default`, and the response says so. `truncated` is true when `max_tokens` ended
the summary rather than the model. Summaries are cached under a hash of the whole
prompt (persona text included), `max_tokens` and `temperature`, so an edited
persona takes effect at once and an identical request returns the same summary. The file and its persona,
plus `max_tokens` for the summary, must fit the model's context (`SUMMARY_N_CTX`,
32,768 tokens by default); otherwise the request is refused with HTTP 413 and the
token counts. Uploads past 5 MB also return 413, and binary files 400.

---

## AST Parsing

### POST /parse-ast

The structure of a Python module: its imports, and its top-level functions,
classes (with their methods) and type aliases. Nested definitions are not listed.

| Form parameter | Description |
| --- | --- |
| `file` | Python source file |
| `language` | Must be `python` |

```bash
curl -X POST "http://localhost:8000/parse-ast" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "language=python"
```

For this module:

```python
"""Pumps."""
from . import valves
from typing import Protocol

type Reading = tuple[str, float]

class Pump(Protocol):
    """A pump."""
    def start(self, rate: float = 1.0) -> None: ...

@register("pumps")
async def drain(tank: str, /, *readings: Reading, limit: int | None = None) -> bool:
    """Empty a tank."""
    valve = Valve(tank)
    return await valve.open()
```

the response is:

```json
{
  "language": "python",
  "docstring": "Pumps.",
  "imports": [".", "typing"],
  "imported_symbols": {"valves": ".", "Protocol": "typing"},
  "functions": [{
    "name": "drain",
    "docstring": "Empty a tank.",
    "params": [
      {"name": "tank", "type": "str", "kind": "positional_only"},
      {"name": "readings", "type": "Reading", "kind": "var_positional"},
      {"name": "limit", "type": "int | None", "kind": "keyword_only", "default": "None"}
    ],
    "returns": "bool",
    "decorators": ["register('pumps')"],
    "is_async": true,
    "body_dependencies": {
      "instantiations": ["Valve"],
      "method_calls": [["valve", "open"]]
    }
  }],
  "classes": [{
    "name": "Pump",
    "docstring": "A pump.",
    "base_classes": ["Protocol"],
    "methods": [{
      "name": "start",
      "docstring": "",
      "params": [
        {"name": "self", "type": "None", "kind": "positional_or_keyword"},
        {"name": "rate", "type": "float", "kind": "positional_or_keyword", "default": "1.0"}
      ],
      "returns": "None",
      "decorators": [],
      "is_async": false,
      "body_dependencies": {"instantiations": [], "method_calls": []}
    }],
    "decorators": []
  }],
  "type_aliases": [{"name": "Reading", "value": "tuple[str, float]"}],
  "exports": ["Reading", "Pump", "drain"]
}
```

| Field | Meaning |
| --- | --- |
| `imports` | Modules imported, sorted; a relative import keeps its dots (`.`, `..pkg`) |
| `imported_symbols` | Each name a `from` import brings in, with its module |
| `params[].type`, `returns` | The annotation as written, or `"None"` where there is none |
| `params[].kind` | `positional_only`, `positional_or_keyword`, `var_positional` (`*args`), `keyword_only` or `var_keyword` (`**kwargs`), in that order |
| `params[].default` | The default as written, when there is one |
| `decorators`, `base_classes` | As written, arguments included |
| `body_dependencies` | Capitalised names the body calls (`instantiations`, each once, in first-use order) and `[object, method]` calls (`method_calls`) |
| `exports` | Top-level functions, classes and type aliases, in source order |

Uploads past 5 MB return HTTP 413. Non-UTF-8 files, Python syntax errors and code
nested too deeply for Python's parser return HTTP 400.

---

## Monitoring

### GET /health

Server status: `"ok"` when every model switched on is loaded, otherwise
`"degraded"` with HTTP 503. Without the key that is all it says:

```bash
curl "http://localhost:8000/health"
```

```json
{"status": "ok"}
```

With the bearer key it adds each model's detail:

```bash
curl "http://localhost:8000/health" -H "Authorization: Bearer $API_KEY"
```

```json
{
  "status": "ok",
  "embedding_model": {
    "name": "google/embeddinggemma-300m",
    "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
    "status": "loaded"
  },
  "summarization_model": {
    "name": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
    "basename": "gemma-4-26B_q4_0-it.gguf",
    "revision": "d1c082be9cf3c8a514acf63b8761f4b41935842e",
    "status": "loaded"
  },
  "amrbart_model": {
    "name": "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2",
    "revision": "12f7e78e3dfa6f37ddbbf77c4c152b4fc39bc512",
    "status": "loaded",
    "device": "mps"
  }
}
```

A model switched off in configuration reports `{"status": "disabled"}`.

### GET /rate-limits

The configured rate-limit budgets. No authentication.

```bash
curl "http://localhost:8000/rate-limits"
```

```json
{
  "embed": {"calls": 5, "seconds": 10},
  "amr": {"calls": 10, "seconds": 60},
  "summarize": {"calls": 2, "seconds": 60}
}
```

Budgets are per client IP, HTTP method, and matched route template, so each
embedding endpoint has its own `embed` budget. Storage is in-process; these
limits assume a single worker; buckets of clients that have gone are swept
away. `/corpus/gpt` and `/parse-ast` are not rate-limited: the first admits one
training and two waiting, and the second runs no model.

---

## Error Responses

| Code | Description |
|------|-------------|
| 400 | Bad request: a binary upload (`/embed`, `/summarize`), a non-UTF-8 one, an unsupported language, or Python that cannot be parsed |
| 401 | Invalid API key |
| 403 | Missing or malformed bearer credentials |
| 413 | A request body or upload past its limit, or a file too long for the summarizer's context |
| 422 | Request validation failed |
| 429 | Too many requests (rate limit exceeded) |
| 499 | The client disconnected before the answer (`/amr/parse-batch`, `/corpus/gpt`) |
| 500 | A server fault; the detail says what failed, and the server log has the rest |
| 503 | A model not loaded (switched off in configuration), `/health` degraded, or `/corpus/gpt` full |

```json
{
  "detail": "Error description"
}
```
