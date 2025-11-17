import ast
import logging

logger = logging.getLogger(__name__)


# Helper to safely extract names from various AST node types
def _get_node_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_get_node_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):  # For default values
        return repr(node.value)
    if node is None:
        return "None"
    return "unknown"


class ASTImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.imported_symbols = {}  # Maps symbol name to module

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
            # Also track specific imported symbols
            for alias in node.names:
                symbol_name = alias.asname if alias.asname else alias.name
                self.imported_symbols[symbol_name] = node.module
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
            obj_name = _get_node_name(node.func.value)
            method_name = node.func.attr
            self.method_calls.append((obj_name, method_name))
        self.generic_visit(node)


def parse_python_code(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Failed to parse Python code due to syntax error: {e}")
        raise e

    import_visitor = ASTImportVisitor()
    import_visitor.visit(tree)
    imports = sorted(list(import_visitor.imports))
    imported_symbols = import_visitor.imported_symbols

    functions = []
    classes = []
    exports = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.append(node.name)
            # --- Extract detailed function info ---
            params = []
            # Align default values with parameters
            defaults = [d for d in node.args.defaults]
            num_defaults = len(defaults)
            param_nodes = node.args.args
            for i, arg in enumerate(param_nodes):
                param_info = {"name": arg.arg, "type": _get_node_name(arg.annotation)}
                # Check if this param has a default value
                if i >= len(param_nodes) - num_defaults:
                    default_index = i - (len(param_nodes) - num_defaults)
                    param_info["default"] = _get_node_name(defaults[default_index])
                params.append(param_info)

            # Extract dependencies from function body
            body_visitor = ASTBodyDependencyVisitor()
            body_visitor.visit(node)

            functions.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "params": params,
                    "returns": _get_node_name(node.returns),
                    "decorators": [_get_node_name(d) for d in node.decorator_list],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "body_dependencies": {
                        "instantiations": list(set(body_visitor.instantiations)),
                        "method_calls": body_visitor.method_calls,
                    },
                }
            )

        elif isinstance(node, ast.ClassDef):
            exports.append(node.name)
            # --- Extract detailed class info ---
            methods = []
            for body_item in node.body:
                if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # (Re-using the same function parsing logic for methods)
                    method_params = [
                        {"name": arg.arg, "type": _get_node_name(arg.annotation)}
                        for arg in body_item.args.args
                    ]

                    # Extract dependencies from method body
                    method_body_visitor = ASTBodyDependencyVisitor()
                    method_body_visitor.visit(body_item)

                    methods.append(
                        {
                            "name": body_item.name,
                            "docstring": ast.get_docstring(body_item) or "",
                            "params": method_params,
                            "returns": _get_node_name(body_item.returns),
                            "decorators": [
                                _get_node_name(d) for d in body_item.decorator_list
                            ],
                            "is_async": isinstance(body_item, ast.AsyncFunctionDef),
                            "body_dependencies": {
                                "instantiations": list(
                                    set(method_body_visitor.instantiations)
                                ),
                                "method_calls": method_body_visitor.method_calls,
                            },
                        }
                    )

            classes.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "base_classes": [_get_node_name(b) for b in node.bases],
                    "methods": methods,
                    "decorators": [_get_node_name(d) for d in node.decorator_list],
                }
            )

    return {
        "language": "python",
        "docstring": ast.get_docstring(tree) or "",
        "imports": imports,
        "imported_symbols": imported_symbols,  # Map of symbol -> module
        "functions": functions,
        "classes": classes,
        "exports": exports,
    }


def parse_code_to_ast(code: str, language: str):
    if language.lower() == "python":
        return parse_python_code(code)
    else:
        raise ValueError(f"Unsupported language for AST parsing: {language}")
