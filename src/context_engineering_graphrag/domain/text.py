"""Text primitives shared by chunking, retrieval, graph extraction and evaluation.

Everything here is deterministic and dependency-free so that ranking,
budgeting and groundedness scores are reproducible across replicas and
test runs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
_PIECE = re.compile(r"[^\W_]+|[^\w\s]|_", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*%?")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "also",
        "may",
        "might",
        "must",
        "shall",
        "within",
        "without",
        "upon",
        "via",
        "per",
    ]
)

NEGATIONS = frozenset({"not", "no", "never", "none", "neither", "nor", "cannot", "without"})


@dataclass(frozen=True, slots=True)
class Span:
    """A slice of a source text with its character offsets."""

    start: int
    end: int
    text: str


class Tokenizer(Protocol):
    """Counts model tokens for budgeting."""

    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """Conservative sub-word token estimate without a vocabulary file.

    Each punctuation mark counts as one token and each word as one token per
    started block of seven characters. Against byte-pair vocabularies used by
    current foundation models this over-estimates slightly, which is the safe
    direction for a context budget. A vocabulary-exact tokenizer can be
    injected through the ``Tokenizer`` protocol.
    """

    def count(self, text: str) -> int:
        total = 0
        for piece in _PIECE.findall(text):
            total += 1 + (len(piece) - 1) // 7 if piece[0].isalnum() else 1
        return total


def normalize(text: str) -> str:
    """Unicode-normalized, case-folded, punctuation-free, single-spaced text."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_WORD.findall(folded))


def words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text))


# (suffix, replacement, minimum stem length, stem needs a vowel, undouble final consonant)
_SUFFIX_RULES: tuple[tuple[str, str, int, bool, bool], ...] = (
    ("ies", "y", 2, False, False),
    ("sses", "ss", 1, False, False),
    ("ingly", "", 3, False, False),
    ("edly", "", 3, False, False),
    ("ing", "", 3, True, True),
    ("ed", "", 3, True, True),
    ("xes", "x", 1, False, False),
    ("zes", "z", 1, False, False),
    ("ses", "s", 1, False, False),
    ("ss", "ss", 0, False, False),
    ("us", "us", 0, False, False),
    ("is", "is", 0, False, False),
    ("s", "", 1, False, False),
)


@lru_cache(maxsize=131_072)
def stem(word: str) -> str:
    """Light, deterministic suffix stripping (S-stemmer plus common verb endings)."""
    w = word.casefold()
    if len(w) <= 3:
        return w
    for suffix, replacement, min_stem, needs_vowel, undouble in _SUFFIX_RULES:
        if not w.endswith(suffix):
            continue
        base = w[: -len(suffix)]
        if len(base) < min_stem or (needs_vowel and not any(c in "aeiouy" for c in base)):
            continue
        if undouble and len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base + replacement
    return w


def terms(text: str) -> list[str]:
    """Index terms: case-folded, stop words removed, stemmed. Order preserved."""
    return [stem(w) for w in (t.casefold() for t in words(text)) if w not in STOPWORDS]


def numbers(text: str) -> frozenset[str]:
    """Numeric literals, normalised so ``1,000`` and ``1000`` compare equal."""
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(text))


def has_negation(text: str) -> bool:
    lowered = text.casefold()
    return any(w in NEGATIONS for w in _WORD.findall(lowered)) or "n't" in lowered


_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "inc",
        "ltd",
        "corp",
        "co",
        "jr",
        "sr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "fig",
        "no",
        "approx",
        "dept",
        "u.s",
        "u.k",
    }
)
_BOUNDARY = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s+|$)|\n\s*\n")


def split_sentences(text: str) -> list[Span]:
    """Sentence spans with offsets into ``text``.

    Splits on terminal punctuation followed by whitespace and on blank lines,
    and does not split after common abbreviations or single initials.
    """
    spans: list[Span] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if match.group().strip(".!?\"'\u201d\u2019)]") == "" and match.group()[0] == ".":
            preceding = text[start : match.start()].split()
            last = preceding[-1].casefold().rstrip(".") if preceding else ""
            if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                continue
        _append_span(text, start, end, spans)
        start = end
    _append_span(text, start, len(text), spans)
    return spans


def _append_span(text: str, start: int, end: int, spans: list[Span]) -> None:
    segment = text[start:end]
    stripped = segment.strip()
    if not stripped:
        return
    offset = start + (len(segment) - len(segment.lstrip()))
    spans.append(Span(offset, offset + len(stripped), stripped))


def shingles(text: str, size: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = [t.casefold() for t in words(text)]
    if len(tokens) < size:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def jaccard(left: frozenset[object] | set[object], right: frozenset[object] | set[object]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0
