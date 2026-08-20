"""Physical circuit structure helpers.

Circuit element suffixes are identifiers, not part of the physical topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CircuitNode:
    kind: str
    value: str | None = None
    children: tuple["CircuitNode", ...] = ()


def _group(kind: str, children: list[CircuitNode]) -> CircuitNode:
    flattened: list[CircuitNode] = []
    for child in children:
        if child.kind == kind:
            flattened.extend(child.children)
        else:
            flattened.append(child)
    if len(flattened) == 1:
        return flattened[0]
    return CircuitNode(kind, children=tuple(flattened))


class _Parser:
    def __init__(self, text: str):
        self.text = re.sub(r"\s+", "", text)
        self.index = 0

    def parse(self) -> CircuitNode:
        if not self.text:
            raise ValueError("circuit is empty")
        node = self._parse_series()
        if self.index != len(self.text):
            raise ValueError(f"unexpected circuit text at position {self.index}")
        return node

    def _parse_series(self) -> CircuitNode:
        children = [self._parse_term()]
        while self.index < len(self.text) and self.text[self.index] == "-":
            self.index += 1
            children.append(self._parse_term())
        return _group("series", children)

    def _parse_term(self) -> CircuitNode:
        if self.text.startswith("p(", self.index):
            self.index += 2
            children = [self._parse_series_until(",", ")")]
            while self.index < len(self.text) and self.text[self.index] == ",":
                self.index += 1
                children.append(self._parse_series_until(",", ")"))
            if self.index >= len(self.text) or self.text[self.index] != ")":
                raise ValueError("unclosed parallel circuit block")
            self.index += 1
            return _group("parallel", children)
        start = self.index
        while self.index < len(self.text) and self.text[self.index] not in "-,)":
            self.index += 1
        if start == self.index:
            raise ValueError(f"expected circuit element at position {self.index}")
        token = self.text[start:self.index]
        element_type = re.match(r"[A-Za-z]+", token)
        if element_type is None:
            raise ValueError(f"invalid circuit element {token!r}")
        return CircuitNode("element", token)

    def _parse_series_until(self, *terminators: str) -> CircuitNode:
        start = self.index
        children = [self._parse_term()]
        while self.index < len(self.text) and self.text[self.index] == "-":
            if self.index + 1 < len(self.text) and self.text[self.index + 1] in terminators:
                break
            self.index += 1
            children.append(self._parse_term())
        if self.index == start:
            raise ValueError("empty circuit block")
        return _group("series", children)


def parse_circuit(circuit: str) -> CircuitNode:
    return _Parser(circuit).parse()


def _element_type(token: str) -> str:
    match = re.match(r"[A-Za-z]+", token)
    return match.group(0).casefold() if match else token.casefold()


def canonical_circuit(circuit: str) -> tuple:
    def canonical(node: CircuitNode) -> tuple:
        if node.kind == "element":
            return ("element", _element_type(node.value or ""))
        children = tuple(sorted((canonical(child) for child in node.children), key=repr))
        return (node.kind, children)

    return canonical(parse_circuit(circuit))


def circuits_equivalent(first: str | None, second: str | None) -> bool:
    try:
        return canonical_circuit(first or "") == canonical_circuit(second or "")
    except ValueError:
        return re.sub(r"\s+", "", str(first or "")).casefold() == re.sub(
            r"\s+", "", str(second or "")
        ).casefold()


def parameter_name_mapping(source: str, target: str) -> dict[str, str] | None:
    """Return source-element-name -> target-element-name correspondence."""
    source_root = parse_circuit(source)
    target_root = parse_circuit(target)
    if canonical_circuit(source) != canonical_circuit(target):
        return None

    mapping: dict[str, str] = {}

    def walk(source_node: CircuitNode, target_node: CircuitNode) -> None:
        if source_node.kind == "element" and target_node.kind == "element":
            mapping[source_node.value or ""] = target_node.value or ""
            return
        source_children = sorted(
            source_node.children,
            key=lambda child: (repr(canonical_circuit_from_node(child)), child.value or ""),
        )
        target_children = sorted(
            target_node.children,
            key=lambda child: (repr(canonical_circuit_from_node(child)), child.value or ""),
        )
        for source_child, target_child in zip(source_children, target_children):
            walk(source_child, target_child)

    def canonical_circuit_from_node(node: CircuitNode) -> tuple:
        if node.kind == "element":
            return ("element", _element_type(node.value or ""))
        return (
            node.kind,
            tuple(sorted((canonical_circuit_from_node(child) for child in node.children), key=repr)),
        )

    walk(source_root, target_root)
    return mapping


def map_parameter_name(name: str, element_mapping: dict[str, str]) -> str | None:
    for source_element, target_element in sorted(element_mapping.items(), key=lambda item: -len(item[0])):
        if name == source_element:
            return target_element
        prefix = source_element + "_"
        if name.startswith(prefix):
            return target_element + name[len(source_element):]
    return None
