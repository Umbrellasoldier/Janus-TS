"""Strict parser for formal MoleCode-TS/v1 generations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Edge

_EDGE_RE = re.compile(
    r"^a(0|[1-9][0-9]*) --\[bo=(0\.5|1|1\.5|2|2\.5|3)\]-- "
    r"a(0|[1-9][0-9]*)$"
)
_OPEN = "<TS_EDGES>"
_CLOSE = "</TS_EDGES>"
_IM_END_TEXT = "<|im_end|>"


@dataclass(frozen=True)
class ParseResult:
    valid: bool
    edges: tuple[Edge, ...]
    error_code: str | None = None
    error_message: str | None = None
    normalized_text: str | None = None


def _invalid(code: str, message: str, normalized: str) -> ParseResult:
    return ParseResult(
        valid=False,
        edges=(),
        error_code=code,
        error_message=message,
        normalized_text=normalized,
    )


def parse_ts_edges(text: str, *, atom_count: int) -> ParseResult:
    """Parse one generation without chemistry repair or forgiving cleanup.

    The only normalization allowed by the frozen protocol is CRLF-to-LF,
    outer whitespace removal, and removal of one terminal chat ``im_end``
    token.  Lines must already be unique, correctly oriented, and sorted.
    """

    if atom_count < 0:
        raise ValueError("atom_count must be non-negative")
    if not isinstance(text, str):
        return _invalid("not_text", "generation is not a string", repr(text))

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalized.endswith(_IM_END_TEXT):
        normalized = normalized[: -len(_IM_END_TEXT)].rstrip()

    lines = normalized.split("\n") if normalized else []
    if len(lines) < 2 or lines[0] != _OPEN or lines[-1] != _CLOSE:
        return _invalid(
            "wrapper",
            "expected exactly one TS_EDGES block and no surrounding text",
            normalized,
        )
    if any(line in {_OPEN, _CLOSE} for line in lines[1:-1]):
        return _invalid("multiple_blocks", "nested or repeated block marker", normalized)

    edges: list[Edge] = []
    last_pair: tuple[int, int] | None = None
    for line_number, line in enumerate(lines[1:-1], start=2):
        match = _EDGE_RE.fullmatch(line)
        if match is None:
            return _invalid(
                "line_syntax",
                f"invalid edge syntax on line {line_number}",
                normalized,
            )
        atom_i = int(match.group(1))
        atom_j = int(match.group(3))
        if atom_i >= atom_j:
            return _invalid("edge_orientation", "every edge requires I < J", normalized)
        if atom_j >= atom_count:
            return _invalid("atom_id", "edge references an unknown atom ID", normalized)
        pair = (atom_i, atom_j)
        if last_pair is not None and pair <= last_pair:
            code = "duplicate_edge" if pair == last_pair else "edge_order"
            return _invalid(code, "edge pairs must be unique and sorted", normalized)
        last_pair = pair
        edges.append(Edge(atom_i, atom_j, float(match.group(2))))

    return ParseResult(valid=True, edges=tuple(edges), normalized_text=normalized)
