"""``calculate`` demo tool (docs/tz/06_llm.md §2).

``get_time``/``get_date`` -- this module's other original TZ-06 demo tools --
moved to ``tools/system_tools.py`` per docs/tz/07_tools.md §7A.2. ``calculate``
stays here since it isn't part of the TZ-07 tool set and the eval harness
(``scripts/eval_llm_tools.py``) imports ``evaluate_expression`` from this
module by name.
"""

from __future__ import annotations

import ast
import operator
from typing import Annotated

from ultron.tools.base import ToolResult, tool

_BINARY_OPS: dict[type[ast.AST], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.AST], object] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ExpressionError(Exception):
    """Raised when ``evaluate_expression`` is given something unsafe/invalid."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        op = _BINARY_OPS[type(node.op)]
        return op(_eval_node(node.left), _eval_node(node.right))  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        op = _UNARY_OPS[type(node.op)]
        return op(_eval_node(node.operand))  # type: ignore[operator]
    raise ExpressionError(f"Ruxsat etilmagan ifoda qismi: {ast.dump(node)}")


def evaluate_expression(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression (+, -, *, /, //, %, **).

    Never uses ``eval()`` -- only a whitelisted AST walk over numeric
    literals and arithmetic operators, so LLM-supplied text can't execute
    arbitrary code.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Ifodani o'qib bo'lmadi: {exc}") from exc
    return _eval_node(tree.body)


@tool(
    name="calculate",
    description=(
        "Evaluate a basic arithmetic expression (+, -, *, /, %, **) and return"
        " the numeric result. Convert the user's words to a plain expression"
        " first, e.g. '17 ni 23 ga ko'paytir' -> '17 * 23'."
    ),
)
async def calculate(
    expression: Annotated[str, "Arithmetic expression as text, e.g. '17 * 23' or '(4+5)/3'"],
) -> ToolResult:
    try:
        result = evaluate_expression(expression)
    except (ExpressionError, ZeroDivisionError, TypeError) as exc:
        return ToolResult(ok=False, error=f"Ifodani hisoblab bo'lmadi: {exc}")

    result_text = str(int(result)) if float(result).is_integer() else f"{result:.6g}"
    return ToolResult(ok=True, data={"result": result}, speak=f"Natija: {result_text}.")
