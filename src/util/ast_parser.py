import ast
import logging

logger = logging.getLogger(__name__)


def _source(node: ast.AST | None) -> str:
    """A node as it is written (annotations, defaults, decorators, bases, call
    targets), or "None" where there is none."""
    if node is None:
        return "None"
    # Dotted names, the common case, are joined once without recursion.
    parts, base = [], node
    while isinstance(base, ast.Attribute):
        parts.append(base.attr)
        base = base.value
    if isinstance(base, ast.Name):
        return ".".join([base.id, *reversed(parts)])
    return ast.unparse(node)


def _param(arg: ast.arg, kind: str, default: ast.expr | None = None) -> dict:
    param = {"name": arg.arg, "type": _source(arg.annotation), "kind": kind}
    if default is not None:
        param["default"] = _source(default)
    return param


def _params(args: ast.arguments) -> list[dict]:
    """Every parameter in order, with its kind: positional-only, then positional
    or keyword, *args, keyword-only and **kwargs."""
    positional = [
        *((arg, "positional_only") for arg in args.posonlyargs),
        *((arg, "positional_or_keyword") for arg in args.args),
    ]
    # Defaults belong to the last positional parameters.
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = [
        _param(arg, kind, default)
        for (arg, kind), default in zip(positional, defaults, strict=True)
    ]
    if args.vararg:
        params.append(_param(args.vararg, "var_positional"))
    params.extend(
        _param(arg, "keyword_only", default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    if args.kwarg:
        params.append(_param(args.kwarg, "var_keyword"))
    return params


class ASTImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.imported_symbols = {}  # Maps symbol name to module

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # A relative import keeps its dots: "from . import x" is ".", and
        # "from ..pkg import y" is "..pkg".
        module = "." * node.level + (node.module or "")
        self.imports.add(module)
        for alias in node.names:
            symbol_name = alias.asname if alias.asname else alias.name
            self.imported_symbols[symbol_name] = module
        self.generic_visit(node)


class ASTBodyDependencyVisitor(ast.NodeVisitor):
    """Extract dependencies from function/method bodies."""

    def __init__(self):
        self.instantiations = []  # List of class names being instantiated
        self.method_calls = []  # List of (object, method) tuples

    def visit_Call(self, node):
        # Check if this is a class instantiation (ClassName())
        if isinstance(node.func, ast.Name):
            # Direct call like: SnowflakeClient()
            if (
                node.func.id and node.func.id[0].isupper()
            ):  # Class names start with uppercase
                self.instantiations.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call like: obj.method()
            self.method_calls.append((_source(node.func.value), node.func.attr))
        self.generic_visit(node)


def _function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    body_visitor = ASTBodyDependencyVisitor()
    body_visitor.visit(node)
    return {
        "name": node.name,
        "docstring": ast.get_docstring(node) or "",
        "params": _params(node.args),
        "returns": _source(node.returns),
        "decorators": [_source(d) for d in node.decorator_list],
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "body_dependencies": {
            # Each class once, in the order the body first names it.
            "instantiations": list(dict.fromkeys(body_visitor.instantiations)),
            "method_calls": body_visitor.method_calls,
        },
    }


def parse_python_code(code: str):
    """The module's imports, and its top-level functions, classes (with their
    methods) and type aliases. Nested definitions are not listed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Failed to parse Python code due to syntax error: {e}")
        raise e

    import_visitor = ASTImportVisitor()
    import_visitor.visit(tree)

    functions = []
    classes = []
    type_aliases = []
    exports = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.append(node.name)
            functions.append(_function(node))
        elif isinstance(node, ast.ClassDef):
            exports.append(node.name)
            classes.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "base_classes": [_source(b) for b in node.bases],
                    "methods": [
                        _function(item)
                        for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                    "decorators": [_source(d) for d in node.decorator_list],
                }
            )
        elif isinstance(node, ast.TypeAlias):
            exports.append(node.name.id)
            type_aliases.append({"name": node.name.id, "value": _source(node.value)})

    return {
        "language": "python",
        "docstring": ast.get_docstring(tree) or "",
        "imports": sorted(import_visitor.imports),
        "imported_symbols": import_visitor.imported_symbols,  # Map of symbol -> module
        "functions": functions,
        "classes": classes,
        "type_aliases": type_aliases,
        "exports": exports,
    }


def parse_code_to_ast(code: str, language: str):
    if language.lower() == "python":
        return parse_python_code(code)
    else:
        raise ValueError(f"Unsupported language for AST parsing: {language}")
