"""/parse-ast reports Python structure as it is written."""

import ast
import json
import os
import subprocess
import sys

from src.util.ast_parser import _source, parse_python_code


def _function(code):
    return parse_python_code(code)["functions"][0]


def test_annotations_and_defaults_are_reported_as_written():
    f = _function("def f(a: list[int], b: int | None = None) -> dict[str, int]: ...")
    assert f["params"] == [
        {"name": "a", "type": "list[int]", "kind": "positional_or_keyword"},
        {
            "name": "b",
            "type": "int | None",
            "kind": "positional_or_keyword",
            "default": "None",
        },
    ]
    assert f["returns"] == "dict[str, int]"


def test_every_kind_of_parameter_is_listed():
    f = _function("def f(a, /, b=1, *args, c, d=2, **kwargs): ...")
    assert [(p["name"], p["kind"], p.get("default")) for p in f["params"]] == [
        ("a", "positional_only", None),
        ("b", "positional_or_keyword", "1"),
        ("args", "var_positional", None),
        ("c", "keyword_only", None),
        ("d", "keyword_only", "2"),
        ("kwargs", "var_keyword", None),
    ]


def test_methods_keep_their_defaults():
    parsed = parse_python_code("class C:\n    def m(self, x: int = 5): ...\n")
    method = parsed["classes"][0]["methods"][0]
    assert method["params"][1] == {
        "name": "x",
        "type": "int",
        "kind": "positional_or_keyword",
        "default": "5",
    }


def test_decorators_with_arguments_are_kept():
    f = _function('@router.post("/x", tags=["a"])\ndef f(): ...')
    assert f["decorators"] == ["router.post('/x', tags=['a'])"]


def test_relative_imports_keep_their_dots():
    parsed = parse_python_code("from . import local\nfrom ..pkg import thing\n")
    assert parsed["imports"] == [".", "..pkg"]
    assert parsed["imported_symbols"] == {"local": ".", "thing": "..pkg"}


def test_type_aliases_are_listed():
    parsed = parse_python_code("type Pair = tuple[int, int]\n")
    assert parsed["type_aliases"] == [{"name": "Pair", "value": "tuple[int, int]"}]
    assert parsed["exports"] == ["Pair"]


def test_instantiations_keep_their_order_whatever_the_hash_seed():
    program = (
        "import json; from src.util.ast_parser import parse_python_code; "
        "src = 'def f():\\n    Alpha()\\n    Beta()\\n    Gamma()\\n    Alpha()\\n'; "
        "print(json.dumps(parse_python_code(src)['functions'][0]"
        "['body_dependencies']['instantiations']))"
    )
    seen = {
        subprocess.check_output(
            [sys.executable, "-c", program],
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "."},
            text=True,
        )
        for seed in ("1", "2", "3", "4", "5")
    }
    assert [json.loads(output) for output in seen] == [["Alpha", "Beta", "Gamma"]]


def test_dotted_names_are_joined_without_recursion():
    """Deeper than the recursion limit, so a recursive rendering could not do it."""
    chain = "a" + "".join(f".b{k}" for k in range(3000))
    node = ast.parse(chain, mode="eval").body
    assert _source(node) == chain
