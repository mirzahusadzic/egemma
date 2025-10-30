# eGemma Personas

This directory contains **provisional, user-controlled personas** for tailoring AI responses to specific contexts.

## Important Notice

⚠️ **These personas are PROVISIONAL and EXPERIMENTAL.**

- **User-controlled**: You can modify, create, or remove personas as needed
- **Not officially endorsed**: These are starting templates, not Google-blessed solutions
- **No security guarantees**: Personas do NOT provide safety or ethical guardrails
- **Your responsibility**: How you use these personas is entirely up to you

## What Are Personas?

Personas are prompt templates that shape AI model responses for specific tasks:

- `code/developer.md` - General code analysis and development
- `code/structural_analyst.md` - AST and structural code analysis
- `code/parser_generator.md` - Tree-sitter parser generation
- `docs/security_validator.md` - Mission document validation
- `docs/operational_validator.md` - Operational pattern analysis
- etc.

## Model Selection

eGemma supports multiple models:

- **Gemma** (local, open source, privacy-focused)
- **Gemini** (API-based, powerful, requires API key)

**The choice of model is yours.** Neither is "more secure" or "officially recommended."
Both Google Gemma and Google Gemini are Google products with their own characteristics:

- Gemma: Local control, privacy, no API costs
- Gemini: Cloud-based, more capable, API costs apply

**You decide which to trust and use.**

## Using Personas

Personas are loaded by the eGemma workbench and passed to models. To use a persona:

```python
# Via API
response = await summarize(
    content=code,
    persona="developer",  # Uses code/developer.md
    model="gemini-2.5-flash"  # Or any supported model
)
```

## Modifying Personas

**You own these files.** To customize:

1. Edit existing personas in `code/` or `docs/`
2. Create new ones following the same format
3. Remove personas you don't want
4. Version control your changes

**No approval needed. No "official" version to follow.**

## Security and Ethics

**Personas do NOT provide:**
- ❌ Content filtering
- ❌ Safety guarantees
- ❌ Ethical enforcement
- ❌ Weaponization prevention

**They are prompt templates**, not security infrastructure.

For cognition-cli users: The dual-use risks of semantic alignment measurement
exist regardless of which personas you use. See `cognition-cli security mandate`
for details.

## License

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
