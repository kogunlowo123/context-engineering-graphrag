"""Text primitives and chunking strategies."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from context_engineering_graphrag.domain.models import ChunkingConfig, ChunkingStrategy
from context_engineering_graphrag.domain.text import (
    HeuristicTokenizer,
    has_negation,
    jaccard,
    normalize,
    numbers,
    shingles,
    split_sentences,
    stem,
    terms,
    words,
)
from context_engineering_graphrag.ingestion.chunking import Chunker

TOK = HeuristicTokenizer()


class TestText:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("studies", "study"),
            ("classes", "class"),
            ("running", "run"),
            ("acquired", "acquir"),
            ("boxes", "box"),
            ("status", "status"),
            ("analysis", "analysis"),
            ("partners", "partner"),
            ("sing", "sing"),
            ("bed", "bed"),
            ("hoping", "hop"),
            ("surprisingly", "surpris"),
            ("the", "the"),
        ],
    )
    def test_stem(self, word: str, expected: str) -> None:
        assert stem(word) == expected

    def test_terms_drop_stopwords_and_stem(self) -> None:
        assert terms("The reagents were shipped to the labs") == ["reagent", "ship", "lab"]

    def test_normalize_and_words(self) -> None:
        assert normalize("  Acme   Diagnostics, Inc. ") == "acme diagnostics inc"
        assert words("the agent\u2019s data") == ["the", "agent\u2019s", "data"]

    def test_tokenizer_counts_long_words_and_punctuation(self) -> None:
        assert TOK.count("a b c") == 3
        assert TOK.count("internationalization!") == 1 + (20 - 1) // 7 + 1
        assert TOK.count("") == 0

    def test_sentence_splitting_respects_abbreviations_and_offsets(self) -> None:
        text = "Dr. Chen leads Acme Inc. today. It grew 5.2% in 2024! Next?\n\nNew paragraph"
        spans = split_sentences(text)
        assert [s.text for s in spans] == [
            "Dr. Chen leads Acme Inc. today.",
            "It grew 5.2% in 2024!",
            "Next?",
            "New paragraph",
        ]
        assert all(text[s.start : s.end] == s.text for s in spans)
        assert split_sentences("   ") == []
        assert [s.text for s in split_sentences("J. Smith arrived. Done.")] == [
            "J. Smith arrived.",
            "Done.",
        ]

    def test_numbers_and_negation(self) -> None:
        assert numbers("Revenue was 1,000 units, up 12% from 900.") == {"1000", "12%", "900"}
        assert has_negation("The audit did not find issues")
        assert has_negation("It isn't approved")
        assert not has_negation("The audit found issues")

    def test_shingles_and_jaccard(self) -> None:
        assert shingles("one two") == {("one", "two")}
        assert shingles("") == frozenset()
        assert len(shingles("a b c d")) == 2
        assert jaccard(set(), set()) == 1.0
        assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


TEXT = (
    "# Supplier overview\n\nAcme Diagnostics depends on Beta Reagents. Beta Reagents is based "
    "in Lyon. The contract renews every two years.\n\n## Logistics\n\nGlobex Logistics ships "
    "reagents across Europe. Port congestion delayed shipments for six weeks. "
    + " ".join(f"Sentence number {i} adds more detail about reagent quality." for i in range(30))
)


class TestChunking:
    @pytest.mark.parametrize("strategy", list(ChunkingStrategy))
    def test_offsets_are_exact_and_limits_hold(self, strategy: ChunkingStrategy) -> None:
        config = ChunkingConfig(strategy=strategy, max_tokens=48, overlap_tokens=8)
        chunks = Chunker(config, TOK).chunk(TEXT)
        assert len(chunks) > 3
        for chunk in chunks:
            assert TEXT[chunk.start : chunk.end] == chunk.text
            assert chunk.token_count <= 48
            assert chunk.text == chunk.text.strip()
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert chunks[0].start == 0
        assert chunks[-1].end == len(TEXT.rstrip())

    @pytest.mark.parametrize("strategy", list(ChunkingStrategy))
    def test_overlap_is_applied(self, strategy: ChunkingStrategy) -> None:
        config = ChunkingConfig(strategy=strategy, max_tokens=40, overlap_tokens=12)
        chunks = Chunker(config, TOK).chunk(TEXT)
        overlaps = sum(1 for a, b in itertools.pairwise(chunks) if b.start < a.end)
        assert overlaps >= len(chunks) // 2

    def test_no_overlap_means_disjoint_chunks(self) -> None:
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE, max_tokens=40, overlap_tokens=0
        )
        chunks = Chunker(config, TOK).chunk(TEXT)
        assert all(b.start >= a.end for a, b in itertools.pairwise(chunks))

    def test_recursive_prefers_section_boundaries(self) -> None:
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE, max_tokens=64, overlap_tokens=0
        )
        chunks = Chunker(config, TOK).chunk(TEXT)
        assert chunks[0].text.startswith("# Supplier overview")
        assert any(c.text.startswith("## Logistics") for c in chunks)

    def test_oversized_sentence_is_split_by_tokens(self) -> None:
        sentence = " ".join(f"word{i}" for i in range(200)) + "."
        for strategy in ChunkingStrategy:
            chunks = Chunker(
                ChunkingConfig(strategy=strategy, max_tokens=32, overlap_tokens=0), TOK
            ).chunk(sentence)
            assert len(chunks) > 5
            assert all(c.token_count <= 32 for c in chunks)

    def test_single_unsplittable_token_still_yields_a_chunk(self) -> None:
        blob = "x" * 600
        chunks = Chunker(
            ChunkingConfig(strategy=ChunkingStrategy.FIXED, max_tokens=32, overlap_tokens=0), TOK
        ).chunk(blob)
        assert [c.text for c in chunks] == [blob]

    def test_empty_text(self) -> None:
        assert Chunker(ChunkingConfig(), TOK).chunk("  \n ") == []

    def test_config_rejects_large_overlap(self) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            ChunkingConfig(max_tokens=64, overlap_tokens=40)

    @settings(max_examples=40, deadline=None)
    @given(
        text=st.text(
            alphabet=st.characters(categories=["L", "N", "P", "Zs"], include_characters="\n"),
            min_size=1,
            max_size=1500,
        ),
        strategy=st.sampled_from(list(ChunkingStrategy)),
    )
    def test_property_chunks_are_faithful_slices(
        self, text: str, strategy: ChunkingStrategy
    ) -> None:
        chunks = Chunker(
            ChunkingConfig(strategy=strategy, max_tokens=40, overlap_tokens=8), TOK
        ).chunk(text)
        covered = "".join(text[c.start : c.end] for c in chunks)
        for chunk in chunks:
            assert text[chunk.start : chunk.end] == chunk.text
        assert set(covered.split()) == set(text.split())
