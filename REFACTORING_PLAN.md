# Egemma Codebase Refactoring Plan

**Date**: 2025-12-14
**Goal**: Improve code organization, maintainability, and scalability
**Estimated Impact**: ~30 files changed, ~200 import updates

---

## Current Problems

1. **Flat structure** - All modules in `src/` root (12+ files)
2. **Generic naming** - `chat.py` doesn't describe its purpose
3. **Monolithic server.py** - 1062 lines, handles 6+ different concerns
4. **Missing rate limiting** - Conversation endpoints unprotected
5. **Unclear boundaries** - OpenAI-specific code mixed with generic code

---

## Target Structure

```
src/
├── models/                    # Model wrappers
│   ├── __init__.py
│   ├── llm.py                 # Renamed from chat.py
│   ├── embedding.py
│   └── summarization.py
│
├── api/                       # API-specific code
│   ├── __init__.py
│   └── openai/                # OpenAI API compatibility
│       ├── __init__.py
│       ├── compat.py          # Renamed from openai_compat.py
│       ├── responses.py       # Response schemas
│       └── conversations.py   # Conversation storage
│
├── routers/                   # FastAPI routers (extracted from server.py)
│   ├── __init__.py
│   ├── embed.py               # /v1/embed endpoint
│   ├── summarize.py           # /v1/summarize endpoint
│   ├── responses.py           # /v1/responses endpoint
│   ├── conversations.py       # /v1/conversations endpoints
│   └── ast_parser.py          # /v1/parse_ast endpoint
│
├── streaming/                 # Streaming utilities
│   ├── __init__.py
│   └── handler.py             # Streaming logic + suppression
│
├── util/                      # Utilities (unchanged)
│   ├── __init__.py
│   ├── rate_limiter.py
│   ├── log_condenser.py
│   ├── file_util.py
│   └── ast_parser.py
│
├── server.py                  # Slim server (app setup + routing)
├── config.py                  # Settings (add conversation rate limits)
└── personas/                  # Unchanged
```

---

## Refactoring Phases

### **Phase 1: Create Folder Structure**
Create new directories without moving files yet.

```bash
mkdir -p src/models
mkdir -p src/api/openai
mkdir -p src/routers
mkdir -p src/streaming
```

**Files to create:**
- `src/models/__init__.py`
- `src/api/__init__.py`
- `src/api/openai/__init__.py`
- `src/routers/__init__.py`
- `src/streaming/__init__.py`

**Verification:** Directories exist, `__init__.py` files present

---

### **Phase 2: Move and Rename Files**

| Current | New Location | Reason |
|---------|--------------|--------|
| `src/chat.py` | `src/models/llm.py` | Better naming, logical grouping |
| `src/embedding.py` | `src/models/embedding.py` | Group model wrappers |
| `src/summarization.py` | `src/models/summarization.py` | Group model wrappers |
| `src/openai_compat.py` | `src/api/openai/compat.py` | OpenAI-specific code |
| `src/responses.py` | `src/api/openai/responses.py` | OpenAI API schemas |
| `src/conversations.py` | `src/api/openai/conversations.py` | OpenAI API feature |

**Commands:**
```bash
git mv src/chat.py src/models/llm.py
git mv src/embedding.py src/models/embedding.py
git mv src/summarization.py src/models/summarization.py
git mv src/openai_compat.py src/api/openai/compat.py
git mv src/responses.py src/api/openai/responses.py
git mv src/conversations.py src/api/openai/conversations.py
```

**Verification:** Files in new locations, git tracks as renames

---

### **Phase 3: Extract Routers from server.py**

Split `server.py` (1062 lines) into focused routers:

#### **3a. Extract /v1/embed → routers/embed.py** (~100 lines)
- Endpoint: `POST /v1/embed`
- Dependencies: models/embedding.py
- Includes: encoder cache, dimension handling

#### **3b. Extract /v1/summarize → routers/summarize.py** (~120 lines)
- Endpoint: `POST /v1/summarize`
- Dependencies: models/summarization.py, util/log_condenser.py
- Includes: summarizer cache, log condensation

#### **3c. Extract /v1/responses → routers/responses.py** (~450 lines)
- Endpoint: `POST /v1/responses`
- Dependencies: models/llm.py, streaming/handler.py
- Includes: streaming logic, tool call parsing

#### **3d. Extract /v1/conversations → routers/conversations.py** (~200 lines)
- Endpoints: All conversation CRUD operations
- Dependencies: api/openai/conversations.py
- Includes: 6 conversation endpoints

#### **3e. Extract /v1/parse_ast → routers/ast_parser.py** (~80 lines)
- Endpoint: `POST /v1/parse_ast`
- Dependencies: util/ast_parser.py
- Includes: AST parsing logic

#### **3f. Create streaming/handler.py** (~150 lines)
- Extract streaming logic from routers/responses.py
- Includes: suppression logic, event generation
- Reusable across endpoints

#### **3g. Update server.py** (target: ~100 lines)
- Keep: App setup, CORS, middleware
- Keep: Router registration
- Remove: All endpoint implementations

**New server.py structure:**
```python
# App setup (~30 lines)
app = FastAPI(...)

# CORS, middleware (~20 lines)
app.add_middleware(...)

# Model initialization (~20 lines)
embedding_model_wrapper = ...
summarization_model_wrapper = ...
chat_model_wrapper = ...

# Router registration (~30 lines)
from .routers import (
    embed_router,
    summarize_router,
    responses_router,
    conversations_router,
    ast_parser_router,
)

app.include_router(embed_router)
app.include_router(summarize_router)
app.include_router(responses_router)
app.include_router(conversations_router)
app.include_router(ast_parser_router)
```

**Verification:** Each router works independently, server.py < 150 lines

---

### **Phase 4: Add Rate Limiting to Conversation Endpoints**

#### **4a. Update config.py**
Add new settings:
```python
# Conversation API rate limiting (generous limits)
CONVERSATION_RATE_LIMIT_SECONDS: int = 60
CONVERSATION_RATE_LIMIT_CALLS: int = 100
```

#### **4b. Apply to routers/conversations.py**
Add rate limiting to all 6 endpoints:
```python
@router.post(
    "/v1/conversations",
    dependencies=[
        Depends(get_api_key),
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
```

#### **4c. Add tests**
- Test rate limiting on conversation endpoints
- Verify 429 errors when limit exceeded

**Verification:** Conversation endpoints return 429 when rate limited

---

### **Phase 5: Update All Imports**

Update imports across **entire codebase**:

#### **Files to update:**
- `tests/test_chat.py` → `tests/test_llm.py` (rename + imports)
- `tests/test_embedding.py` → update imports
- `tests/test_summarization.py` → update imports
- `tests/test_responses.py` → update imports
- `tests/test_conversations.py` → update imports
- `tests/test_server.py` → update imports
- `tests/test_openai_compat.py` → update imports
- All `src/` files with cross-imports

#### **Import mapping:**
```python
# Old → New
from src.chat import ChatModelWrapper
→ from src.models.llm import ChatModelWrapper

from src.responses import Response
→ from src.api.openai.responses import Response

from src.conversations import ConversationManager
→ from src.api.openai.conversations import ConversationManager

from src.openai_compat import create_openai_chat_completion
→ from src.api.openai.compat import create_openai_chat_completion
```

**Tools to use:**
- Grep for old imports
- Edit files with new imports
- Use `git grep` to find missed imports

**Verification:** No import errors, all tests discover modules

---

### **Phase 6: Run Full Test Suite**

#### **6a. Run pytest**
```bash
cd /Users/MHUSADZI/src/egemma
uv run pytest -v --cov=src --cov-report=term-missing
```

**Expected:**
- All 185 tests pass
- Coverage ≥ 87%
- No import errors

#### **6b. Run linting**
```bash
uv run ruff check .
```

**Expected:**
- No linting errors
- All imports resolve

#### **6c. Fix any issues**
- Fix import errors
- Fix test failures
- Update test file names if needed

**Verification:** All tests pass, coverage maintained, no linting errors

---

### **Phase 7: Documentation and Commit**

#### **7a. Update README.md**
- Document new structure
- Update import examples
- Add architecture diagram

#### **7b. Update CHANGELOG.md**
Add entry for refactoring:
```markdown
## [Unreleased]
### Changed
- **BREAKING**: Refactored codebase structure
  - Renamed `chat.py` → `models/llm.py`
  - Moved OpenAI-specific code to `api/openai/`
  - Split `server.py` into focused routers
  - Added rate limiting to conversation endpoints
```

#### **7c. Create migration guide**
Document import changes for users:
```markdown
# Migration Guide: v0.x → v1.0

## Import Changes
- `from src.chat import ...` → `from src.models.llm import ...`
- `from src.responses import ...` → `from src.api.openai.responses import ...`
```

#### **7d. Commit strategy**
Create commits for each phase:
1. `refactor: create new folder structure`
2. `refactor: move and rename files to new structure`
3. `refactor: extract routers from server.py`
4. `feat: add rate limiting to conversation endpoints`
5. `refactor: update all imports for new structure`
6. `docs: update README and migration guide for refactoring`

**Verification:** Git history is clean, changes are reviewable

---

## Testing Strategy

### **Unit Tests**
- All existing tests must pass
- Update test imports
- Rename test files to match new structure

### **Integration Tests**
- Test that routers work independently
- Test that rate limiting works on all endpoints
- Test that streaming still works correctly

### **Manual Testing**
- Start server: `uv run uvicorn src.server:app`
- Test each endpoint manually
- Verify rate limiting with rapid requests

---

## Rollback Plan

If refactoring fails:
1. **Git reset**: `git reset --hard HEAD~N` (N = number of commits)
2. **Branch protection**: Work on `refactor` branch, merge only after success
3. **Backup**: Tag current state before refactoring: `git tag pre-refactor`

---

## Success Criteria

- ✅ All 185+ tests pass
- ✅ Coverage ≥ 87%
- ✅ No linting errors
- ✅ Server starts without errors
- ✅ All endpoints functional
- ✅ Rate limiting works on conversations
- ✅ Import structure is clear and logical
- ✅ `server.py` < 150 lines
- ✅ Documentation updated

---

## Estimated Effort

| Phase | Time | Complexity |
|-------|------|------------|
| Phase 1 | 5 min | Low |
| Phase 2 | 10 min | Low |
| Phase 3 | 60 min | High |
| Phase 4 | 15 min | Medium |
| Phase 5 | 30 min | Medium |
| Phase 6 | 20 min | Medium |
| Phase 7 | 15 min | Low |
| **Total** | **~2.5 hours** | **Medium-High** |

---

## Notes

- This is a **breaking change** for anyone importing modules directly
- Recommend doing this in one session to avoid partial state
- Consider creating `refactor` branch first
- Tag current state: `git tag pre-refactor`
