"""Chunking strategies.

All strategies return chunks with exact character offsets into the source
text, so every chunk can be traced back to (and re-verified against) the
original document, and overlap is expressed in model tokens rather than
characters.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from context_engineering_graphrag.domain.models import ChunkingConfig, ChunkingStrategy
from context_engineering_graphrag.domain.text import Span, Tokenizer, split_sentences

_PIECE = re.compile(r"\S+")
_SEPARATORS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n(?=#{1,6}\s)"),  # markdown section headings
    re.compile(r"\n[ \t]*\n"),  # paragraphs
    re.compile(r"\n"),  # lines
)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    start: int
    end: int
    text: str
    token_count: int


class Chunker:
    def __init__(self, config: ChunkingConfig, tokenizer: Tokenizer) -> None:
        self._config = config
        self._tok = tokenizer

    def chunk(self, text: str) -> list[ChunkDraft]:
        if not text.strip():
            return []
        match self._config.strategy:
            case ChunkingStrategy.FIXED:
                spans = self._fixed(text, 0, len(text), overlap=self._config.overlap_tokens)
            case ChunkingStrategy.SENTENCE:
                spans = self._sentence(text)
            case ChunkingStrategy.RECURSIVE:
                spans = self._with_word_overlap(text, self._recursive(text, 0, len(text), 0))
        return [
            ChunkDraft(i, s, e, text[s:e], self._tok.count(text[s:e]))
            for i, (s, e) in enumerate(spans)
        ]

    # ----------------------------------------------------------------- fixed

    def _fixed(self, text: str, start: int, end: int, *, overlap: int) -> list[tuple[int, int]]:
        pieces = [(m.start() + start, m.end() + start) for m in _PIECE.finditer(text[start:end])]
        costs = [self._tok.count(text[s:e]) for s, e in pieces]
        limit = self._config.max_tokens
        spans: list[tuple[int, int]] = []
        first = 0
        while first < len(pieces):
            used = 0
            last = first
            while last < len(pieces) and (used + costs[last] <= limit or last == first):
                used += costs[last]
                last += 1
            spans.append((pieces[first][0], pieces[last - 1][1]))
            if last >= len(pieces):
                break
            back = last
            carried = 0
            while back - 1 > first and carried + costs[back - 1] <= overlap:
                back -= 1
                carried += costs[back]
            first = back
        return spans

    # -------------------------------------------------------------- sentence

    def _sentence(self, text: str) -> list[tuple[int, int]]:
        units: list[Span] = []
        for sentence in split_sentences(text):
            if self._tok.count(sentence.text) <= self._config.max_tokens:
                units.append(sentence)
            else:
                units.extend(
                    Span(s, e, text[s:e])
                    for s, e in self._fixed(text, sentence.start, sentence.end, overlap=0)
                )
        costs = [self._tok.count(u.text) for u in units]
        limit = self._config.max_tokens
        spans: list[tuple[int, int]] = []
        first = 0
        while first < len(units):
            used = 0
            last = first
            while last < len(units) and (used + costs[last] <= limit or last == first):
                used += costs[last]
                last += 1
            spans.append((units[first].start, units[last - 1].end))
            if last >= len(units):
                break
            back = last
            carried = 0
            while (
                back - 1 > first
                and carried + costs[back - 1] <= self._config.overlap_tokens
                and used - carried - costs[back - 1] > 0
            ):
                back -= 1
                carried += costs[back]
            first = back
        return spans

    # ------------------------------------------------------------- recursive

    def _recursive(self, text: str, start: int, end: int, depth: int) -> list[tuple[int, int]]:
        segment = text[start:end]
        if self._tok.count(segment) <= self._config.max_tokens:
            return [_trim(text, start, end)] if segment.strip() else []
        if depth >= len(_SEPARATORS):
            sentences = list(split_sentences(segment))
            if len(sentences) > 1:
                parts = [(start + s.start, start + s.end) for s in sentences]
                return self._merge(text, parts, depth=len(_SEPARATORS) + 1)
            return self._fixed(text, start, end, overlap=0)
        cuts = [m.start() + start for m in _SEPARATORS[depth].finditer(segment)]
        if not cuts:
            return self._recursive(text, start, end, depth + 1)
        bounds = [start, *cuts, end]
        parts = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        return self._merge(text, parts, depth=depth + 1)

    def _merge(
        self, text: str, parts: list[tuple[int, int]], *, depth: int
    ) -> list[tuple[int, int]]:
        """Greedily join adjacent parts up to the limit; split oversize parts deeper."""
        out: list[tuple[int, int]] = []
        current: tuple[int, int] | None = None
        for part_start, part_end in parts:
            if not text[part_start:part_end].strip():
                continue
            if self._tok.count(text[part_start:part_end]) > self._config.max_tokens:
                if current is not None:
                    out.append(_trim(text, *current))
                    current = None
                if depth > len(_SEPARATORS):
                    out.extend(self._fixed(text, part_start, part_end, overlap=0))
                else:
                    out.extend(self._recursive(text, part_start, part_end, depth))
                continue
            if current is None:
                current = (part_start, part_end)
            elif self._tok.count(text[current[0] : part_end]) <= self._config.max_tokens:
                current = (current[0], part_end)
            else:
                out.append(_trim(text, *current))
                current = (part_start, part_end)
        if current is not None:
            out.append(_trim(text, *current))
        return out

    def _with_word_overlap(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Extend each chunk backwards over the previous chunk by up to overlap tokens."""
        overlap = self._config.overlap_tokens
        if overlap == 0 or len(spans) < 2:
            return spans
        result = [spans[0]]
        for (prev_start, prev_end), (start, end) in itertools.pairwise(spans):
            own = self._tok.count(text[start:end])
            budget = min(overlap, self._config.max_tokens - own)
            new_start = start
            carried = 0
            words = [
                (m.start() + prev_start, m.end() + prev_start)
                for m in _PIECE.finditer(text[prev_start:prev_end])
            ]
            for word_start, word_end in reversed(words):
                cost = self._tok.count(text[word_start:word_end])
                if carried + cost > budget:
                    break
                carried += cost
                new_start = word_start
            result.append((new_start, end))
        return result


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    segment = text[start:end]
    left = len(segment) - len(segment.lstrip())
    right = len(segment) - len(segment.rstrip())
    return start + left, end - right
