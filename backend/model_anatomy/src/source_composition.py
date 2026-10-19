"""Small source-defined composition patterns, independent of model and attribute names."""

from __future__ import annotations

import ast
import inspect
import textwrap

import torch

from .wrapper_proof import forward_wrappers


def body(module: torch.nn.Module) -> list[ast.stmt]:
    """Read a bounded undecorated/native-passthrough body without invoking the module."""
    if "forward" in vars(module) or module._forward_hooks or module._forward_pre_hooks:
        return []
    if (inspect.getattr_static(type(module), "__getattribute__", object.__getattribute__) is not object.__getattribute__
            or inspect.getattr_static(type(module), "__getattr__", torch.nn.Module.__getattr__) is not torch.nn.Module.__getattr__):
        return []
    function = type(module).forward
    if forward_wrappers(function, module, {})["status"] == "unknown":
        return []
    text = textwrap.dedent(inspect.getsource(inspect.unwrap(function)))
    if len(text) > 65536:
        return []
    tree = ast.parse(text)
    if sum(1 for _ in ast.walk(tree)) > 2048:
        return []
    function_ast = tree.body[0]
    if not isinstance(function_ast, ast.FunctionDef):
        return []
    positional = function_ast.args.posonlyargs + function_ast.args.args
    if not positional or positional[0].arg != "self":
        return []
    return [node for node in function_ast.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and type(node.value.value) is str)]


def assignment(node: ast.stmt) -> tuple[str, ast.expr] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    return None


def child_call(node: ast.expr, module: torch.nn.Module) -> tuple[str, torch.nn.Module] | None:
    if (isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1
            and isinstance(node.args[0], ast.Name) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
        child = module._modules.get(node.func.attr)
        if child is not None and not child._forward_hooks and not child._forward_pre_hooks and "forward" not in vars(child):
            return node.args[0].id, child
    return None


def projection_terminal(module: torch.nn.Module) -> bool:
    if type(module) is torch.nn.Linear:
        return True
    statements = body(module)
    if len(statements) < 2 or not isinstance(statements[-1], ast.Return):
        return False
    last = assignment(statements[-2])
    if last is None or not isinstance(statements[-1].value, ast.Name) or statements[-1].value.id != last[0]:
        return False
    call = child_call(last[1], module)
    return call is not None and type(call[1]) is torch.nn.Linear


def source_composition(module: torch.nn.Module) -> tuple[str, str] | None:
    """Recognize explicit selection/projection or independent returned projection branches."""
    try:
        statements = body(module)
        if not statements or not isinstance(statements[-1], ast.Return):
            return None
        returned = statements[-1].value
        assignments = [assignment(node) for node in statements[:-1]]
        if not assignments or any(item is None for item in assignments):
            return None
        first_name, first_value = assignments[0]
        if (len(assignments) in {2, 3} and isinstance(first_value, ast.Subscript)
                and isinstance(first_value.value, ast.Name) and isinstance(first_value.slice, ast.Tuple)
                and len(first_value.slice.elts) == 2):
            batch, position = first_value.slice.elts
            if (isinstance(batch, ast.Slice) and batch.lower is None and batch.upper is None and batch.step is None
                    and isinstance(position, ast.Constant) and type(position.value) is int):
                previous = first_name
                for index, (name, value) in enumerate(assignments[1:]):
                    call = child_call(value, module)
                    allowed = (torch.nn.Linear,) if index == 0 else (torch.nn.Tanh, torch.nn.GELU, torch.nn.ReLU, torch.nn.SiLU, torch.nn.Sigmoid)
                    if call is None or call[0] != previous or type(call[1]) not in allowed:
                        break
                    previous = name
                else:
                    if isinstance(returned, ast.Name) and returned.id == previous:
                        return "selection_projection", "Explicit second-axis element selection followed by native affine projection and optional native activation"
        if isinstance(returned, ast.Tuple) and len(returned.elts) >= 2 and len(returned.elts) == len(assignments):
            branch_names = [name for name, _ in assignments]
            calls = [child_call(value, module) for _, value in assignments]
            if (all(isinstance(value, ast.Name) for value in returned.elts)
                    and [value.id for value in returned.elts] == branch_names
                    and len(set(branch_names)) == len(branch_names)
                    and all(call is not None for call in calls)
                    and len({id(call[1]) for call in calls}) == len(calls)
                    and all(call[0] not in branch_names and projection_terminal(call[1]) for call in calls)):
                return "output_head_group", "Independent source input branches terminate in native affine projections and are returned as distinct tuple outputs; task intent is not inferred"
    except (OSError, TypeError, ValueError, AttributeError, SyntaxError, RecursionError):
        return None
    return None
