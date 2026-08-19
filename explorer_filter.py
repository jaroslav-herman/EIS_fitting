"""Shared filtering for the fit and DRT parameter explorers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


NUMERIC_OPERATORS = ("=", "!=", "<", "<=", ">", ">=")
TEXT_OPERATORS = ("=", "!=", "contains", "does not contain")


@dataclass
class FilterCondition:
    field: str
    operator: str
    value: str


@dataclass
class FilterDefinition:
    conditions: list[FilterCondition] = field(default_factory=list)
    match: str = "all"

    @property
    def active(self) -> bool:
        return bool(self.conditions)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
        try:
            return math.isnan(float(value))
        except ValueError:
            return False
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def field_is_numeric(records: Iterable[dict[str, object]], field: str) -> bool:
    values = [
        record.get(field)
        for record in records
        if not _is_missing(record.get(field))
    ]
    return bool(values) and all(_number(value) is not None for value in values)


def field_operators(records: Iterable[dict[str, object]], field: str) -> tuple[str, ...]:
    return NUMERIC_OPERATORS if field_is_numeric(records, field) else TEXT_OPERATORS


def _matches(value: object, condition: FilterCondition, numeric: bool) -> bool:
    if _is_missing(value):
        return False
    if numeric:
        left = _number(value)
        right = _number(condition.value)
        if left is None or right is None:
            return False
        return {
            "=": left == right,
            "!=": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[condition.operator]
    left = str(value).casefold()
    right = condition.value.casefold()
    if condition.operator == "=":
        return left == right
    if condition.operator == "!=":
        return left != right
    if condition.operator == "contains":
        return right in left
    if condition.operator == "does not contain":
        return right not in left
    return False


def apply_filters(
    records: Iterable[dict[str, object]], definition: FilterDefinition | None
) -> list[dict[str, object]]:
    records = list(records)
    if definition is None or not definition.conditions:
        return records
    field_types = {
        condition.field: field_is_numeric(records, condition.field)
        for condition in definition.conditions
    }
    filtered = []
    for record in records:
        matches = [
            condition.field in record
            and _matches(record.get(condition.field), condition, field_types[condition.field])
            for condition in definition.conditions
        ]
        keep = all(matches) if definition.match == "all" else any(matches)
        if keep:
            filtered.append(record)
    return filtered
