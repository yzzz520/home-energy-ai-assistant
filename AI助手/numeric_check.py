from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class NumericAudit:
    valid: bool
    unmatched_numbers: list[str]
    allowed_count: int


def _collect_numbers(value: Any, output: set[float]) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            output.add(float(value))
        return
    if isinstance(value, str):
        for token in NUMBER_PATTERN.findall(value):
            output.add(float(token))
        return
    if isinstance(value, BaseModel):
        _collect_numbers(value.model_dump(mode="json"), output)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, output)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_numbers(item, output)


def _tolerance(token: str) -> float:
    if "." not in token:
        return 1e-9
    decimal_places = len(token.split(".", 1)[1])
    return 0.5 * 10 ** (-decimal_places) + 1e-9


def check_answer_numbers(
    answer: str,
    tool_payloads: list[Any],
) -> NumericAudit:
    allowed_numbers: set[float] = set()
    for payload in tool_payloads:
        _collect_numbers(payload, allowed_numbers)

    unmatched: list[str] = []
    for match in NUMBER_PATTERN.finditer(answer):
        token = match.group(0)
        value = float(token)
        tolerance = _tolerance(token)
        if not any(
            abs(value - allowed) <= tolerance
            for allowed in allowed_numbers
        ):
            unmatched.append(token)
    return NumericAudit(
        valid=not unmatched,
        unmatched_numbers=unmatched,
        allowed_count=len(allowed_numbers),
    )
