from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any, Mapping, Sequence


getcontext().prec = 40

_NUMBER_RE = re.compile(
    r"(?P<number>\d[\d.\s]*(?:,\d+)?)\s*"
    r"(?P<unit>tỷ|triệu|nghìn|ngàn|%|đồng)?",
    flags=re.IGNORECASE,
)
_MULTIPLIERS = {
    "tỷ": Decimal("1000000000"),
    "triệu": Decimal("1000000"),
    "nghìn": Decimal("1000"),
    "ngàn": Decimal("1000"),
    "đồng": Decimal("1"),
    "": Decimal("1"),
}


class CalculationError(ValueError):
    pass


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized, "f")
    return format(normalized, "f").rstrip("0").rstrip(".")


def parse_legal_numbers(text: object) -> list[Decimal]:
    """Parse all common Vietnamese numbers in their textual order."""

    source = str(text or "").strip().casefold()
    matches = list(_NUMBER_RE.finditer(source))
    if not matches:
        raise CalculationError(f"Không đọc được số từ nguồn: {text!r}")

    values: list[Decimal] = []
    for match in matches:
        raw = re.sub(r"[\s.]", "", match.group("number")).replace(",", ".")
        try:
            number = Decimal(raw)
        except InvalidOperation as exc:
            raise CalculationError(f"Số không hợp lệ: {text!r}") from exc
        unit = (match.group("unit") or "").casefold()
        if unit == "%":
            values.append(number / Decimal("100"))
        else:
            values.append(number * _MULTIPLIERS[unit])
    return values


def parse_legal_number(text: object, occurrence: int = 1) -> Decimal:
    """Parse the 1-based numeric occurrence from a Vietnamese legal span."""

    if occurrence < 1:
        raise CalculationError("occurrence phải lớn hơn hoặc bằng 1")
    values = parse_legal_numbers(text)
    if occurrence > len(values):
        raise CalculationError(
            f"source_text chỉ có {len(values)} số, không có số thứ {occurrence}"
        )
    return values[occurrence - 1]


def _evaluate_node(node: ast.AST, variables: Mapping[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, variables)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise CalculationError(f"Biến chưa được khai báo: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub)
    ):
        value = _evaluate_node(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, variables)
        right = _evaluate_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise CalculationError("Không thể chia cho 0")
            return left / right
    raise CalculationError("Biểu thức chỉ được dùng +, -, *, / và ngoặc")


def safe_decimal_eval(
    expression: str,
    variables: Mapping[str, Decimal],
) -> Decimal:
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        raise CalculationError("Biểu thức không hợp lệ") from exc
    return _evaluate_node(tree, variables)


def _source_corpus(
    *,
    question: str,
    evidences: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    sources = {"question": str(question or "")}
    for index, evidence in enumerate(evidences, 1):
        sources[f"evidence_{index}"] = str(
            evidence.get("noi_dung_goc")
            or evidence.get("provision_content")
            or ""
        )
    return sources


def execute_calculation_plan(
    plan: Mapping[str, Any],
    *,
    question: str,
    evidences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Validate provenance and execute calculations proposed by Agent 5.

    Every variable needs a continuous source_text in the question, a numbered
    evidence, or a previously verified calculation. Its normalized numeric
    value must agree with that source.
    """

    sources = _source_corpus(question=question, evidences=evidences)
    verified: list[dict[str, Any]] = []
    derived_sources: dict[str, dict[str, Any]] = {}
    for calculation_index, calculation in enumerate(
        plan.get("calculations", []) or [], 1
    ):
        if not isinstance(calculation, Mapping):
            continue
        variables: dict[str, Decimal] = {}
        provenance: list[dict[str, Any]] = []
        try:
            for item in calculation.get("variables", []) or []:
                name = str(item.get("name") or "").strip()
                value_text = str(item.get("value") or "").strip()
                source = str(item.get("source") or "").strip().casefold()
                source_text = str(item.get("source_text") or "").strip()
                occurrence = int(item.get("occurrence") or 1)
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                    raise CalculationError(f"Tên biến không hợp lệ: {name!r}")
                declared = Decimal(value_text)
                if source in sources:
                    normalized_span = re.sub(r"\s+", " ", source_text).casefold()
                    normalized_source = re.sub(
                        r"\s+", " ", sources[source]
                    ).casefold()
                    if not normalized_span or normalized_span not in normalized_source:
                        raise CalculationError(
                            f"Không xác minh được source_text của biến {name}"
                        )
                    extracted = parse_legal_number(source_text, occurrence)
                elif source in derived_sources:
                    derived = derived_sources[source]
                    if source_text != str(derived["label"]):
                        matching_sources = [
                            (candidate_source, candidate)
                            for candidate_source, candidate in derived_sources.items()
                            if source_text == str(candidate["label"])
                        ]
                        if len(matching_sources) != 1:
                            raise CalculationError(
                                f"source_text của biến {name} phải đúng nhãn "
                                f"phép tính trước: {derived['label']}"
                            )
                        source, derived = matching_sources[0]
                    extracted = derived["value"]
                    occurrence = 1
                elif re.fullmatch(r"calculation_\d+", source):
                    raise CalculationError(
                        f"Nguồn {source} chưa được xác minh hoặc nằm sau phép tính hiện tại"
                    )
                else:
                    raise CalculationError(f"Nguồn không tồn tại: {source}")
                tolerance = max(abs(extracted) * Decimal("0.000001"), Decimal("0.000001"))
                if abs(declared - extracted) > tolerance:
                    raise CalculationError(
                        f"Giá trị biến {name} không khớp nguồn"
                    )
                variables[name] = declared
                provenance.append(
                    {
                        "name": name,
                        "value": _decimal_text(declared),
                        "source": source,
                        "source_text": source_text,
                        "occurrence": occurrence,
                    }
                )

            expression = str(calculation.get("expression") or "").strip()
            result = safe_decimal_eval(expression, variables)
            label = str(calculation.get("label") or "Phép tính")
            verified.append(
                {
                    "label": label,
                    "expression": expression,
                    "variables": provenance,
                    "result": _decimal_text(result),
                    "unit": str(calculation.get("unit") or "").strip(),
                }
            )
            derived_sources[f"calculation_{calculation_index}"] = {
                "label": label,
                "value": result,
            }
        except (CalculationError, InvalidOperation, TypeError, ValueError) as exc:
            verified.append(
                {
                    "label": str(calculation.get("label") or "Phép tính"),
                    "error": str(exc),
                }
            )
    return verified
