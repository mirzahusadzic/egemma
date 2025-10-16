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

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
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

    functions = []
    classes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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

            functions.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "params": params,
                    "returns": _get_node_name(node.returns),
                    "decorators": [_get_node_name(d) for d in node.decorator_list],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
            )

        elif isinstance(node, ast.ClassDef):
            # --- Extract detailed class info ---
            methods = []
            for body_item in node.body:
                if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # (Re-using the same function parsing logic for methods)
                    method_params = [
                        {"name": arg.arg, "type": _get_node_name(arg.annotation)}
                        for arg in body_item.args.args
                    ]
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
        "functions": functions,
        "classes": classes,
    }


def parse_code_to_ast(code: str, language: str):
    if language.lower() == "python":
        return parse_python_code(code)
    else:
        raise ValueError(f"Unsupported language for AST parsing: {language}")
