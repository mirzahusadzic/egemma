# Repository guide

eGemma is a local inference service: embeddings, a corpus-trained transformer,
AMR parsing, summarization, judgment and Python AST parsing behind one FastAPI
app. Prefer focused fixes over new endpoints, frameworks or dependencies.

---

### Map

The README is the brief; `docs/api.md` is the full contract of every endpoint.

| Task | Source |
| --- | --- |
| App, authentication, `/health`, `/rate-limits`, router wiring | `src/server.py` |
| Settings, read from `.env` or the environment | `src/config.py` |
| Embeddings | `src/routers/embed.py`, `src/models/embedding.py` |
| Corpus transformer, adapted from nanoGPT | `src/routers/corpus.py`, `src/models/corpus_gpt.py` |
| AMR | `src/routers/amr.py`, `src/models/amrbart.py` |
| Summarization and its personas | `src/routers/summarize.py`, `src/models/summarization.py`, `personas/` |
| Judgment and its schema checks | `src/routers/judge.py`, `src/models/judge_schema.py`, `src/models/summarization.py` |
| AST parsing | `src/routers/ast_parser.py`, `src/util/ast_parser.py` |
| Rate and body limits, result cache, log condensing | `src/util/` |
| Adapted code and model licenses | `THIRD_PARTY_NOTICES.md`, `LICENSES/` |

---

### Preserve these behaviors

- Local only: no request content goes to a hosted API. Models load from pinned
  revisions, and each model's results name its revision.
- An input past a model's window is refused or reported with an explicit
  status, never cut without saying so; `/embed/texts` truncates, as documented.
- `/corpus/gpt` and `/judge` admit one request running and two waiting, then
  answer 503 with `Retry-After`. The body is validated before admission, and a
  slot is given back on every outcome.
- A server fault is logged and answered with an opaque 500, never as a client
  error.
- `/judge` schemas are bounded before llama.cpp sees a grammar; schemas that
  crashed it stay covered by tests run in a child process.
- One worker: rate limits and admission live in one process.

---

### Validate and document

- Checks: `uv run ruff check .` and `uv run pytest`, which needs 80% coverage.
  Tests mock inference, download no models and use dummy credentials; CI
  (`.github/workflows/ci.yml`) runs both.
- A behavior change or a fix comes with a test that fails on the old code and
  passes after.
- Dependencies are pinned to exact versions in `requirements.txt`.
- Docs stay generic: they describe eGemma's own contract, not any caller's use.
  Docs-only changes need link and fact checks, not a full test run.
- Docs shape: top-level sections are `###` headings under a `---` rule, since
  GitHub underlines `#` and `##`; section leads of two or three sentences at
  most; tables only for tabular data, run-in lists otherwise; explanation
  before a code block, no comments inside; at most one alert per section; each
  link once per page.
- Commits: a Conventional Commits subject of at most 72 characters, a blank
  line, then a body hard-wrapped at 72; check with
  `git log --format=%B origin/main..main | awk 'length > 72'`.
