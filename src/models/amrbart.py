"""Local text-to-AMR inference with the pinned AMRBART parser."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import penman
import torch
from penman.models.amr import model as amr_model

from ..config import settings

if TYPE_CHECKING:
    from transformers import BartForConditionalGeneration


_POINTER_PATTERN = re.compile(r"<pointer:(\d+)>")
_SYMBOL_PATTERN = re.compile(r'[^\s()":/<>~#]+')
_ROLE_PATTERN = re.compile(r':[^\s()":/<>~#]+')
_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"')
_DISPLAY_TOKEN_PATTERN = re.compile(
    r'<lit>.*?</lit>|"(?:\\.|[^"\\])*"|[()]|[^\s()]+', re.DOTALL
)
_DIAGNOSTIC_TOKEN_PATTERN = re.compile(
    r'<lit>.*?(?:</lit>|$)|"(?:\\.|[^"\\])*"|[()]|[^\s()]+', re.DOTALL
)


def _join_split_roles(tokens: list[str]) -> list[str]:
    """Join detached colons for AMRBART's open-ended op/prep role tokens.

    Tokenization keeps quoted and checkpoint literals intact. No graph node,
    participant, or relation name is invented; the full decoder still validates
    the result. Exact model bytes remain in linearized_raw.
    """
    joined: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            tokens[index] == ":"
            and index + 1 < len(tokens)
            and re.fullmatch(
                r"(?:op[1-9]\d*|prep-[a-z]+(?:-[a-z]+)*)", tokens[index + 1]
            )
        ):
            joined.append(":" + tokens[index + 1])
            index += 2
        else:
            joined.append(tokens[index])
            index += 1
    return joined


class AMRDecodeError(ValueError):
    """Raised when AMRBART emits pointer notation that cannot form an AMR graph."""


@dataclass(frozen=True)
class AMRTriple:
    """One ordered triple from a validated PENMAN graph."""

    source: str
    relation: str
    target: str
    target_kind: str


@dataclass(frozen=True)
class AMRFragment:
    """One mechanically recoverable relation from an invalid graph attempt."""

    source: str | None
    relation: str
    target: str | None
    target_kind: str
    token_start: int
    token_end: int
    complete: bool


@dataclass(frozen=True)
class AMRParseResult:
    """One attributed parser return, before downstream semantic interpretation."""

    model: str
    revision: str
    device: str
    input_tokens: int
    output_tokens: int
    linearized: str
    penman: str | None
    top: str | None
    triples: tuple[AMRTriple, ...]
    fragments: tuple[AMRFragment, ...]
    decode_status: str
    decode_error: str | None = None
    linearized_raw: str | None = None


def format_linearized_amr(linearized: str) -> str:
    """Indent pointer notation for display, leaving the raw model text untouched.

    Call this only after ``pointer_graph_to_penman`` has validated the graph.
    Literal spans remain one token so parentheses inside them are not treated
    as graph structure.
    """

    value = linearized.strip()
    if value.startswith("<AMR>"):
        value = value[len("<AMR>") :].lstrip()
    if value.endswith("</AMR>"):
        value = value[: -len("</AMR>")].rstrip()
    tokens = _join_split_roles(_DISPLAY_TOKEN_PATTERN.findall(value))

    def node(index: int, depth: int, prefix: str = "") -> tuple[list[str], int]:
        if index + 2 >= len(tokens) or tokens[index] != "(":
            raise AMRDecodeError("cannot format pointer graph node")
        lines = [f"{'  ' * depth}{prefix}({tokens[index + 1]} {tokens[index + 2]}"]
        index += 3
        while index < len(tokens) and tokens[index] != ")":
            role = tokens[index]
            if not role.startswith(":") or index + 1 >= len(tokens):
                raise AMRDecodeError("cannot format pointer graph relation")
            index += 1
            if tokens[index] == "(":
                child, index = node(index, depth + 1, f"{role} ")
                lines.extend(child)
            else:
                lines.append(f"{'  ' * (depth + 1)}{role} {tokens[index]}")
                index += 1
        if index >= len(tokens):
            raise AMRDecodeError("cannot format unclosed pointer graph")
        lines[-1] += ")"
        return lines, index + 1

    lines, end = node(0, 0)
    if end != len(tokens):
        raise AMRDecodeError("cannot format trailing pointer graph tokens")
    return "\n".join(["<AMR>", *lines, "</AMR>"])


def pointer_graph_to_penman(linearized: str) -> str:
    """Validate and normalize AMRBART pointer notation as PENMAN.

    AMRBART declares a node as ``(<pointer:n> concept ...)`` and later reuses
    ``<pointer:n>`` as a reference. This decoder performs only that mechanical
    lowering, including detached colons on op/prep role tokens. It does not
    supply missing concepts, reinterpret relations, or replace model output.
    """

    value = linearized.strip()
    if value.startswith("<AMR>"):
        value = value[len("<AMR>") :].lstrip()
    if value.endswith("</AMR>"):
        value = value[: -len("</AMR>")].rstrip()

    # AMRBART represents graph string constants with checkpoint-specific
    # ``<lit>...</lit>`` tokens. Lower those tokens mechanically to a quoted
    # PENMAN string before parsing; their contents are data, not graph syntax.
    tokens = []
    for token in _DISPLAY_TOKEN_PATTERN.findall(value):
        if token.startswith("<lit>"):
            if not token.endswith("</lit>"):
                raise AMRDecodeError("AMRBART returned an unterminated literal")
            token = json.dumps(token[5:-6].strip(), ensure_ascii=False)
        tokens.append(token)
    tokens = _join_split_roles(tokens)
    if not tokens:
        raise AMRDecodeError("AMRBART returned an empty graph")

    lowered: list[str] = []
    declarations: set[str] = set()
    references: set[str] = set()
    def node(index: int) -> int:
        if index >= len(tokens) or tokens[index] != "(":
            raise AMRDecodeError("expected a pointer graph node")
        if index + 1 >= len(tokens):
            raise AMRDecodeError("missing pointer declaration")
        pointer_match = _POINTER_PATTERN.fullmatch(tokens[index + 1])
        if pointer_match is None:
            raise AMRDecodeError("expected a pointer declaration after '('")
        pointer = pointer_match.group(1)
        if pointer in declarations:
            raise AMRDecodeError(f"pointer {pointer} is declared more than once")
        if index + 2 >= len(tokens) or not _SYMBOL_PATTERN.fullmatch(tokens[index + 2]):
            raise AMRDecodeError(f"pointer {pointer} has no concept")
        declarations.add(pointer)
        lowered.extend(["(", f"z{pointer}", "/", tokens[index + 2]])
        index += 3

        while index < len(tokens) and tokens[index] != ")":
            role = tokens[index]
            if not _ROLE_PATTERN.fullmatch(role):
                raise AMRDecodeError("expected a relation after the node concept")
            lowered.append(role)
            index += 1
            if index >= len(tokens):
                raise AMRDecodeError(f"relation {role} has no target")
            token = tokens[index]
            if token == "(":
                index = node(index)
                continue
            reference = _POINTER_PATTERN.fullmatch(token)
            if reference:
                references.add(reference.group(1))
                lowered.append(f"z{reference.group(1)}")
            elif _SYMBOL_PATTERN.fullmatch(token) or _STRING_PATTERN.fullmatch(token):
                lowered.append(token)
            else:
                raise AMRDecodeError(f"relation {role} has no valid target")
            index += 1

        if index >= len(tokens):
            raise AMRDecodeError("unclosed pointer graph")
        lowered.append(")")
        return index + 1

    # PENMAN deliberately accepts missing operands and only consumes the first
    # graph. Validate the entire generated structure before handing it over.
    if node(0) != len(tokens):
        raise AMRDecodeError("trailing tokens after the pointer graph")

    missing = sorted(references - declarations, key=int)
    if missing:
        raise AMRDecodeError("references undeclared pointer(s): " + ", ".join(missing))

    candidate = " ".join(lowered)
    try:
        graph = penman.decode(candidate, model=amr_model)
    except Exception as exc:  # penman exposes several parse exception classes
        raise AMRDecodeError(f"invalid generated graph: {exc}") from exc

    return penman.encode(graph, model=amr_model)


def scan_invalid_amr_fragments(linearized: str) -> tuple[AMRFragment, ...]:
    """Retain complete token-local relations without repairing an invalid graph.

    The scanner recognizes pointer declarations, relation tokens, pointer
    references, and checkpoint literals. It does not balance, resolve, or
    validate the graph; every result therefore remains diagnostic-only.
    """

    matches = list(_DIAGNOSTIC_TOKEN_PATTERN.finditer(linearized))
    tokens = [match.group(0) for match in matches]
    stack: list[str | None] = []
    fragments: list[AMRFragment] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            source = stack[-1] if stack else None
            if index + 2 < len(tokens):
                pointer = _POINTER_PATTERN.fullmatch(tokens[index + 1])
                concept = tokens[index + 2]
                if (
                    pointer
                    and concept not in {"(", ")"}
                    and not concept.startswith(":")
                ):
                    source = f"z{pointer.group(1)}"
                    fragments.append(
                        AMRFragment(
                            source=source,
                            relation=":instance",
                            target=concept,
                            target_kind="literal",
                            token_start=matches[index + 1].start(),
                            token_end=matches[index + 2].end(),
                            complete=True,
                        )
                    )
            stack.append(source)
            index += 1
            continue
        if token == ")":
            if stack:
                stack.pop()
            index += 1
            continue
        if token.startswith(":"):
            source = stack[-1] if stack else None
            target: str | None = None
            target_kind = "unknown"
            token_end = matches[index].end()
            complete = False
            next_index = index + 1
            if next_index < len(tokens):
                next_token = tokens[next_index]
                if next_token == "(" and next_index + 1 < len(tokens):
                    pointer = _POINTER_PATTERN.fullmatch(tokens[next_index + 1])
                    if pointer:
                        target = f"z{pointer.group(1)}"
                        target_kind = "node"
                        token_end = matches[next_index + 1].end()
                        complete = True
                else:
                    pointer = _POINTER_PATTERN.fullmatch(next_token)
                    if pointer:
                        target = f"z{pointer.group(1)}"
                        target_kind = "node"
                        token_end = matches[next_index].end()
                        complete = True
                    elif next_token.startswith("<lit>"):
                        literal = next_token
                        complete = literal.endswith("</lit>")
                        target = (
                            literal.removeprefix("<lit>").removesuffix("</lit>").strip()
                        )
                        target_kind = "literal"
                        token_end = matches[next_index].end()
                    elif next_token not in {"(", ")"} and not next_token.startswith(
                        ":"
                    ):
                        target = next_token
                        target_kind = "literal"
                        token_end = matches[next_index].end()
                        complete = True
            fragments.append(
                AMRFragment(
                    source=source,
                    relation=token,
                    target=target,
                    target_kind=target_kind,
                    token_start=matches[index].start(),
                    token_end=token_end,
                    complete=complete,
                )
            )
        index += 1
    return tuple(fragments)


def _select_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AMRBART_DEVICE=cuda, but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("AMRBART_DEVICE=mps, but MPS is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AMRBartModelWrapper:
    """Load and query AMRBART without its legacy training framework."""

    def __init__(self) -> None:
        self.model: BartForConditionalGeneration | None = None
        self.tokenizer = None
        self.device = "not_loaded"
        self._inference_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load_model(self) -> None:
        from transformers import BartForConditionalGeneration, BartTokenizer

        # Matching the checkpoint's declared tokenizer class avoids AutoTokenizer
        # attempting to import the authors' complete 2022 training repository.
        runtime_tokenizer = type("AMRBartTokenizer", (BartTokenizer,), {})
        self.tokenizer = runtime_tokenizer.from_pretrained(
            settings.AMRBART_MODEL_NAME,
            revision=settings.AMRBART_MODEL_REVISION,
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            settings.AMRBART_MODEL_NAME,
            revision=settings.AMRBART_MODEL_REVISION,
        )

        embedding_count = self.model.get_input_embeddings().num_embeddings
        if embedding_count != len(self.tokenizer):
            raise RuntimeError(
                "AMRBART tokenizer/model vocabulary mismatch: "
                f"{len(self.tokenizer)} != {embedding_count}"
            )

        self.device = _select_device(settings.AMRBART_DEVICE)
        self.model.to(self.device)
        self.model.eval()

    def parse(self, text: str) -> AMRParseResult:
        if not self.is_loaded:
            raise RuntimeError("AMRBART model not loaded")

        assert self.model is not None
        tokenizer = self.tokenizer
        amr_bos = tokenizer.convert_tokens_to_ids("<AMR>")
        amr_eos = tokenizer.convert_tokens_to_ids("</AMR>")
        if amr_bos == tokenizer.unk_token_id or amr_eos == tokenizer.unk_token_id:
            raise RuntimeError("AMRBART tokenizer lacks graph boundary tokens")

        encoded = tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0].tolist()
        # Reproduce the checkpoint's unified text-to-AMR input contract:
        # ``<s> text </s> <AMR> <mask> </AMR>``.
        input_ids.extend([amr_bos, tokenizer.mask_token_id, amr_eos])
        if len(input_ids) > settings.AMRBART_MAX_SOURCE_TOKENS:
            # Keep one attributed result per batch item without running inference
            # on an incomplete source. No graph was generated for this input.
            return AMRParseResult(
                model=settings.AMRBART_MODEL_NAME,
                revision=settings.AMRBART_MODEL_REVISION,
                device=self.device,
                input_tokens=len(input_ids),
                output_tokens=0,
                linearized="",
                linearized_raw="",
                penman=None,
                top=None,
                triples=(),
                fragments=(),
                decode_status="input_too_long",
                decode_error=(
                    f"Input requires {len(input_ids)} source tokens "
                    "including boundaries; "
                    f"limit is {settings.AMRBART_MAX_SOURCE_TOKENS}. "
                    "Split the source into shorter inputs."
                ),
            )
        source = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(source)

        with self._inference_lock, torch.inference_mode():
            generated = self.model.generate(
                input_ids=source,
                attention_mask=attention_mask,
                num_beams=settings.AMRBART_NUM_BEAMS,
                max_length=settings.AMRBART_MAX_GRAPH_TOKENS,
                min_length=0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                decoder_start_token_id=amr_bos,
                eos_token_id=amr_eos,
                do_sample=False,
            )

        token_ids = generated[0].detach().cpu().tolist()
        linearized = tokenizer.decode(token_ids, skip_special_tokens=False)
        try:
            normalized = pointer_graph_to_penman(linearized)
        except AMRDecodeError as exc:
            return AMRParseResult(
                model=settings.AMRBART_MODEL_NAME,
                revision=settings.AMRBART_MODEL_REVISION,
                device=self.device,
                input_tokens=len(input_ids),
                output_tokens=len(token_ids),
                linearized=linearized,
                linearized_raw=linearized,
                penman=None,
                top=None,
                triples=(),
                fragments=scan_invalid_amr_fragments(linearized),
                decode_status="invalid",
                decode_error=str(exc),
            )

        graph = penman.decode(normalized, model=amr_model)
        try:
            displayed_linearized = format_linearized_amr(linearized)
        except AMRDecodeError:
            # A display-only formatter must not invalidate an otherwise valid
            # graph or replace the exact model return retained below.
            displayed_linearized = linearized
        variables = graph.variables()
        triples = tuple(
            AMRTriple(
                source=str(source),
                relation=str(relation),
                target=str(target),
                target_kind="node" if target in variables else "literal",
            )
            for source, relation, target in graph.triples
        )
        return AMRParseResult(
            model=settings.AMRBART_MODEL_NAME,
            revision=settings.AMRBART_MODEL_REVISION,
            device=self.device,
            input_tokens=len(input_ids),
            output_tokens=len(token_ids),
            linearized=displayed_linearized,
            linearized_raw=linearized,
            penman=normalized,
            top=graph.top,
            triples=triples,
            fragments=(),
            decode_status="ok",
        )
