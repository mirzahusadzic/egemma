# eGemma Personas

This directory contains **provisional, user-controlled personas** for tailoring AI responses to specific contexts.

---

### Important Notice

⚠️ **These personas are PROVISIONAL and EXPERIMENTAL.**

- **User-controlled**: You can modify, create, or remove personas as needed
- **Not officially endorsed**: These are starting templates, not Google-blessed solutions
- **No security guarantees**: Personas do NOT provide safety or ethical guardrails
- **Your responsibility**: How you use these personas is entirely up to you

---

### What Are Personas?

Personas are prompt templates that shape AI model responses for specific tasks:

- `code/developer.md` - General code analysis and development
- `code/structural_analyst.md` - AST and structural code analysis
- `code/parser_generator.md` - Tree-sitter parser generation
- `docs/security_validator.md` - Mission document validation
- `docs/operational_validator.md` - Operational pattern analysis
- etc.

---

### Model

Personas are system prompts for the local summarization model, Gemma 4 by
default (`SUMMARY_MODEL_NAME` in `src/config.py`). Content never leaves the
machine. Any instruction-tuned GGUF configured there receives the same personas.

---

### Using Personas

Pass a persona's file name to `/summarize`:

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $WORKBENCH_API_KEY" \
  -F "file=@code.py" \
  -F "persona=developer"
```

- **Folders:** Markdown files use `docs/`; every other file type uses `code/`. A
  persona that does not exist falls back to that folder's `default.md`.
- **Placeholders:** `{language}` and `{max_tokens}` are filled in. Any other
  brace, such as in a JSON example, is kept as written; doubled braces (`{{`,
  `}}`) stand for single ones.

---

### Modifying Personas

**You own these files.** To customize:

1. Edit existing personas in `code/` or `docs/`
2. Create new ones following the same format
3. Remove personas you don't want
4. Version control your changes

An edit takes effect on the next request, without a restart. Each summary names
the persona file it used, so a fallback to `default.md` is visible.

**No approval needed. No "official" version to follow.**

---

### Security and Ethics

**Personas do NOT provide:**
- ❌ Content filtering
- ❌ Safety guarantees
- ❌ Ethical enforcement
- ❌ Weaponization prevention

**They are prompt templates**, not security infrastructure.

For cognition-cli users: The dual-use risks of semantic alignment measurement
exist regardless of which personas you use. See `cognition-cli security mandate`
for details.

---

### License

These personas are provided as-is under the MIT license (same as eGemma).

**NO WARRANTY. NO LIABILITY.** Use at your own risk.

You are solely responsible for:
- How you modify these personas
- Which models you choose to use
- How you deploy AI-assisted tools
- The consequences of your deployment choices

---

**Version**: 1.0
**Status**: Provisional, User-Controlled
**Last Updated**: 2025-10-30
