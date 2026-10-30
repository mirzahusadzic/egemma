"""The caller's JSON schema for /judge: checked, turned into the llama.cpp grammar
that holds the answer, and used again to validate the answer the grammar let
through, since the grammar holds only part of JSON Schema."""

import itertools
import json
import re
import time
from typing import Any

import llama_cpp
import regex
from jsonschema import Draft202012Validator, ValidationError, validators
from jsonschema.exceptions import SchemaError
from llama_cpp import LlamaGrammar

from ..config import settings


class JudgeSchemaError(ValueError):
    """A schema the answer cannot be held to, or one /judge does not take."""


# Bounds the converter expands one level or symbol per unit: maxLength 50,000
# nests a rule 50,000 deep, and llama.cpp's parser recurses once per level, past
# the end of a thread's stack.
_REPEAT_KEYWORDS = ("minLength", "maxLength", "minItems", "maxItems")
_QUANTIFIER = re.compile(r"\{(\d+)(?:,(\d*))?\}")  # {m}, {m,} and {m,n}
# Keywords whose values map names to schemas, and those whose values are data.
_NAME_MAPS = ("properties", "$defs", "definitions", "dependentSchemas")
_DATA = ("enum", "const", "default", "examples")
# The grammar ignores these, and validating them would run the caller's regular
# expressions through Python's re, which cannot be stopped.
_REFUSED = ("patternProperties", "propertyNames")
# A backstop past the bounds above: grammar nesting this deep still parses on an
# 8 MB stack (10,000 does).
_MAX_GRAMMAR_DEPTH = 4096
# All of one answer's pattern checks together, and the violations reported.
_PATTERN_SECONDS = 0.5
_MAX_VIOLATIONS = 20


def _bound(keyword: str, value: int) -> int:
    limit = settings.JUDGE_MAX_SCHEMA_REPEAT
    if value > limit:
        raise JudgeSchemaError(f"{keyword} {value} is past the limit of {limit}.")
    return value


def check_judge_schema(schema: Any) -> None:
    """Refuse a schema /judge cannot hold an answer to safely. It costs a walk of
    the schema, so it runs before a request is admitted."""
    if not isinstance(schema, dict):
        raise JudgeSchemaError("The schema is a JSON object.")
    repeats = 0
    nodes = 0
    stack: list[tuple[Any, int, str]] = [(schema, 0, "schema")]
    while stack:
        node, depth, kind = stack.pop()
        nodes += 1  # every node takes a byte or more of the schema's JSON
        if nodes > settings.JUDGE_MAX_SCHEMA_BYTES:
            raise JudgeSchemaError(
                f"The schema is larger than {settings.JUDGE_MAX_SCHEMA_BYTES} bytes."
            )
        if depth > settings.JUDGE_MAX_SCHEMA_DEPTH:
            raise JudgeSchemaError(
                f"The schema nests deeper than {settings.JUDGE_MAX_SCHEMA_DEPTH}."
            )
        if isinstance(node, list):
            child_kind = "data" if kind == "data" else "schema"
            stack.extend((child, depth + 1, child_kind) for child in node)
            continue
        if not isinstance(node, dict):
            continue
        # The converter resolves a $ref wherever it stands, data included, and
        # refuses an https reference only by an assert, which python -O removes.
        ref = node.get("$ref")
        if ref is not None and not (isinstance(ref, str) and ref.startswith("#")):
            raise JudgeSchemaError("A $ref may only point inside the schema.")
        if kind != "schema":  # names or data: the keys are not keywords
            stack.extend(
                (child, depth + 1, kind if kind == "data" else "schema")
                for child in node.values()
            )
            continue
        for keyword in _REFUSED:
            if keyword in node:
                raise JudgeSchemaError(
                    f"{keyword} is not supported: the grammar cannot hold it, and "
                    "checking it after could take unbounded time."
                )
        for keyword in _REPEAT_KEYWORDS:
            value = node.get(keyword)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                repeats += _bound(keyword, int(value))
        pattern = node.get("pattern")
        if isinstance(pattern, str):
            for low, high in _QUANTIFIER.findall(pattern):
                repeats += sum(
                    _bound("A pattern's {m,n}", int(b)) for b in (low, high) if b
                )
        properties = node.get("properties")
        if isinstance(properties, dict) and (
            len(properties) > settings.JUDGE_MAX_SCHEMA_PROPERTIES
        ):
            raise JudgeSchemaError(
                f"An object has {len(properties)} properties, past the limit of "
                f"{settings.JUDGE_MAX_SCHEMA_PROPERTIES}."
            )
        for key, child in node.items():
            child_kind = (
                "names" if key in _NAME_MAPS else "data" if key in _DATA else "schema"
            )
            stack.append((child, depth + 1, child_kind))
    if repeats > settings.JUDGE_MAX_SCHEMA_REPEAT_TOTAL:
        raise JudgeSchemaError(
            f"The schema's length and count bounds add up to {repeats}, past the "
            f"limit of {settings.JUDGE_MAX_SCHEMA_REPEAT_TOTAL}."
        )
    if len(json.dumps(schema, separators=(",", ":"))) > settings.JUDGE_MAX_SCHEMA_BYTES:
        raise JudgeSchemaError(
            f"The schema is larger than {settings.JUDGE_MAX_SCHEMA_BYTES} bytes."
        )
    try:
        validators.validator_for(schema, default=Draft202012Validator).check_schema(
            schema
        )
    except SchemaError as error:
        raise JudgeSchemaError(f"The schema is not valid: {error.message}") from error


def _grammar_depth(grammar: str) -> int:
    """How deep the grammar's groups nest, outside literals and character classes."""
    depth = deepest = 0
    quote = escape = False
    in_class = False
    for char in grammar:
        if escape:
            escape = False
        elif char == "\\" and (quote or in_class):
            escape = True
        elif quote:
            quote = char != '"'
        elif in_class:
            in_class = char != "]"
        elif char == '"':
            quote = True
        elif char == "[":
            in_class = True
        elif char == "(":
            depth += 1
            deepest = max(deepest, depth)
        elif char == ")":
            depth -= 1
    return deepest


def judge_grammar(schema: Any) -> LlamaGrammar:
    """The grammar that holds an answer to `schema`, once llama.cpp has parsed it."""
    check_judge_schema(schema)
    try:
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
    except Exception as error:  # the converter asserts; any failure is the schema's
        raise JudgeSchemaError(f"The schema cannot be held to: {error}") from error
    if _grammar_depth(grammar._grammar) > _MAX_GRAMMAR_DEPTH:
        raise JudgeSchemaError("The schema's grammar nests too deeply to parse.")
    # llama-cpp-python adds a grammar sampler that llama.cpp failed to build to
    # the chain unchecked, and the first sample then crashes the process; so
    # llama.cpp parses the grammar here first. A schema's grammar names no
    # tokens, so the parse needs no vocabulary.
    sampler = llama_cpp.llama_sampler_init_grammar(
        None, grammar._grammar.encode("utf-8"), grammar._root.encode("utf-8")
    )
    if not sampler:
        raise JudgeSchemaError(
            "llama.cpp cannot parse the schema's grammar: its repetitions exceed "
            "llama.cpp's limit."
        )
    llama_cpp.llama_sampler_free(sampler)
    return grammar


def _pointer(path) -> str:
    return "".join(
        "/" + str(part).replace("~", "~0").replace("/", "~1") for part in path
    )


def answer_violations(schema: dict, answer: Any) -> list[dict[str, str]]:
    """Where `answer` breaks `schema`, as JSON pointers and messages: the grammar
    holds the answer's shape, but not minimum, maximum, uniqueItems and the
    like. A pattern is checked by `regex` within one shared time budget."""
    deadline = time.monotonic() + _PATTERN_SECONDS

    def pattern(validator, pattern, instance, schema):
        if not validator.is_type(instance, "string"):
            return
        try:
            found = regex.search(
                pattern, instance, timeout=max(deadline - time.monotonic(), 0.001)
            )
        except TimeoutError:
            yield ValidationError(f"{pattern!r} could not be checked in time")
            return
        if not found:
            yield ValidationError(f"{instance!r} does not match {pattern!r}")

    base = validators.validator_for(schema, default=Draft202012Validator)
    checker = validators.extend(base, {"pattern": pattern})(schema)
    errors = itertools.islice(checker.iter_errors(answer), _MAX_VIOLATIONS)
    return [
        {"path": _pointer(error.absolute_path), "message": error.message}
        for error in errors
    ]
