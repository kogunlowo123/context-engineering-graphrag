"""Knowledge-graph algorithms."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from agentic_core.domain.classification import DataClassification

from context_engineering_graphrag.domain.models import Assertion, Community, Entity, Relation
from context_engineering_graphrag.graph.community import build_graph, louvain, modularity
from context_engineering_graphrag.graph.extraction import (
    LOCATION,
    ORGANIZATION,
    PERSON,
    PatternEntityExtractor,
    ground_extraction,
)
from context_engineering_graphrag.graph.resolution import (
    EntityResolver,
    NamedItem,
    acronym,
    blocking_keys,
    core_name,
    jaro_winkler,
    name_similarity,
)
from context_engineering_graphrag.graph.search import (
    GraphView,
    aggregate_relations,
    local_search,
    personalized_pagerank,
    phrase,
    rank_communities,
)
from context_engineering_graphrag.graph.summarize import (
    CommunityFacts,
    ExtractiveCommunitySummarizer,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2025, 1, 1, tzinfo=UTC)
T2 = datetime(2026, 1, 1, tzinfo=UTC)


class TestPatternExtraction:
    async def test_entities_types_and_relations(self) -> None:
        text = (
            "Acme Corp acquired Beta Labs in March 2024. Dr. Jane Smith leads Beta Labs, which is "
            "based in Boston. Beta Labs was acquired by Northwind Holdings. The FDA approved the "
            "assay. Bank of America invested in Contoso Freight."
        )
        result = await PatternEntityExtractor().extract(text)
        names = {e.surface: e.entity_type for e in result.entities}
        assert names["Acme Corp"] == ORGANIZATION
        assert names["Jane Smith"] == PERSON
        assert names["Boston"] == LOCATION
        assert names["FDA"] == ORGANIZATION
        assert "Bank of America" in names
        assert "March" not in names
        assert "The" not in names
        for entity in result.entities:
            assert text[entity.start : entity.end] == entity.surface
        triples = {(r.source, r.relation_type, r.target) for r in result.relations}
        assert ("Acme Corp", "acquired", "Beta Labs") in triples
        assert ("Jane Smith", "leads", "Beta Labs") in triples
        assert ("Beta Labs", "located_in", "Boston") in triples
        assert ("Northwind Holdings", "acquired", "Beta Labs") in triples  # passive reversed
        assert ("Bank of America", "invested_in", "Contoso Freight") in triples

    async def test_cooccurrence_fallback_and_gazetteer(self) -> None:
        extractor = PatternEntityExtractor({"orion analyzer": "product", "hplc": "method"})
        result = await extractor.extract("The orion analyzer and Globex Logistics share HPLC data.")
        types = {e.surface: e.entity_type for e in result.entities}
        assert types["orion analyzer"] == "product"
        assert types["HPLC"] == "method"
        assert any(r.relation_type == "related_to" and r.weight == 0.5 for r in result.relations)

    def test_grounding_drops_hallucinations(self) -> None:
        text = "Acme Corp acquired Beta Labs. Beta Labs is based in Lyon."
        raw = (
            'Sure! {"entities": [{"name": "Acme Corp", "type": "organization"},'
            '{"name": "Beta Labs", "type": "organization"},'
            '{"name": "Initech", "type": "organization"}, "bad", {"name": ""}],'
            '"relations": [{"source": "Acme Corp", "target": "Beta Labs", "type": '
            '"Acquired!", "evidence": "Acme Corp acquired Beta Labs."},'
            '{"source": "Acme Corp", "target": "Initech", "type": "owns", '
            '"evidence": "Acme Corp acquired Beta Labs."},'
            '{"source": "Beta Labs", "target": "Acme Corp", "type": "owns", '
            '"evidence": "Beta Labs owns Acme."}, 7]}'
        )
        result = ground_extraction(text, raw)
        assert [e.surface for e in result.entities] == ["Acme Corp", "Beta Labs"]
        assert [(r.source, r.relation_type, r.target) for r in result.relations] == [
            ("Acme Corp", "acquired_", "Beta Labs")
        ]
        assert ground_extraction(text, "no json here").entities == ()
        assert ground_extraction(text, "{not json}").entities == ()
        assert ground_extraction(text, "[1, 2]").entities == ()


class TestResolution:
    def test_names(self) -> None:
        assert core_name("Globex Corporation") == "globex"
        assert core_name("The Globex Group Inc.") == "globex"
        assert acronym("International Business Machines") == "ibm"
        assert acronym("IBM") == "ibm"
        assert acronym("Acme") is None
        assert {"a:ibm", "c:international business machines"} <= blocking_keys(
            "International Business Machines"
        )

    def test_jaro_winkler_reference_values(self) -> None:
        assert jaro_winkler("martha", "marhta") == pytest.approx(0.9611, abs=1e-4)
        assert jaro_winkler("dixon", "dicksonx") == pytest.approx(0.8133, abs=1e-4)
        assert jaro_winkler("abc", "abc") == 1.0
        assert jaro_winkler("", "abc") == 0.0
        assert jaro_winkler("abc", "xyz") == 0.0

    def test_similarity_rules(self) -> None:
        assert name_similarity("Globex Corp", "Globex Corporation") == 1.0
        assert name_similarity("IBM", "International Business Machines") == 0.92
        assert name_similarity("Acme Labs", "Acme Holdings") < 0.88
        assert name_similarity("Northwind Holdings", "Northwnd Holdings") > 0.9
        assert name_similarity("Inc", "Inc") == 1.0

    def test_cluster_and_type_compatibility(self) -> None:
        resolver = EntityResolver()
        items = [
            NamedItem("e1", "IBM", "organization", 5),
            NamedItem("e2", "International Business Machines", "organization", 2),
            NamedItem("e3", "Globex Corp", "entity", 1),
            NamedItem("e4", "Globex Corporation", "organization", 3),
            NamedItem("e5", "Jordan", "person", 1),
            NamedItem("e6", "Jordan", "location", 1),
        ]
        clusters = [[i.key for i in c] for c in resolver.cluster(items)]
        assert ["e1", "e2"] in clusters
        assert ["e4", "e3"] in clusters
        assert ["e5"] in clusters
        assert ["e6"] in clusters
        assert resolver.best_match(NamedItem("", "Globex", "entity"), items).key == "e4"  # type: ignore[union-attr]
        assert resolver.best_match(NamedItem("", "Initech", "entity"), items) is None
        with pytest.raises(ValueError, match="threshold"):
            EntityResolver(0.2)


class TestCommunities:
    def two_cliques(self) -> dict[str, dict[str, float]]:
        edges = [(a, b, 1.0) for a in "abcd" for b in "abcd" if a < b]
        edges += [(a, b, 1.0) for a in "wxyz" for b in "wxyz" if a < b]
        edges.append(("d", "w", 0.5))
        return build_graph(edges, nodes=["lonely"])

    def test_louvain_recovers_planted_partition(self) -> None:
        graph = self.two_cliques()
        levels = louvain(graph)
        final = levels[-1]
        assert final["a"] == final["b"] == final["c"] == final["d"]
        assert final["w"] == final["x"] == final["y"] == final["z"]
        assert final["a"] != final["w"]
        assert final["lonely"] not in {final["a"], final["w"]}
        assert modularity(graph, final) > 0.4
        assert modularity(graph, dict.fromkeys(graph, 0)) == pytest.approx(0.0)

    def test_hierarchy_coarsens(self) -> None:
        # Four triangles in two pairs: finer level has four communities, coarser two.
        edges: list[tuple[str, str, float]] = []
        for group in ("abc", "def", "ghi", "jkl"):
            edges += [(x, y, 3.0) for x in group for y in group if x < y]
        edges += [
            ("c", "d", 1.0),
            ("i", "j", 1.0),
            ("b", "e", 1.0),
            ("h", "k", 1.0),
            ("f", "g", 0.1),
        ]
        levels = louvain(build_graph(edges))
        assert len({*levels[0].values()}) >= len({*levels[-1].values()})
        assert all(set(level) == set("abcdefghijkl") for level in levels)

    def test_degenerate_graphs(self) -> None:
        assert louvain({}) == []
        assert louvain(build_graph([], nodes=["a", "b"])) == [{"a": 0, "b": 1}]
        assert modularity({"a": {}}, {"a": 0}) == 0.0
        with pytest.raises(ValueError, match="positive"):
            build_graph([("a", "b", 0.0)])
        assert build_graph([("a", "a", 1.0)]) == {"a": {}}


def entity(eid: str, name: str, mentions: int = 1, aliases: tuple[str, ...] = ()) -> Entity:
    from context_engineering_graphrag.domain.text import normalize

    return Entity(
        entity_id=eid,
        collection="c",
        name=name,
        normalized=normalize(name),
        entity_type="organization",
        aliases=aliases,
        mention_count=mentions,
    )


def assertion(
    source: str, target: str, kind: str, start: datetime, end: datetime | None, *, n: int = 0
) -> Assertion:
    return Assertion(
        assertion_id=f"a{source}{target}{kind}{n}",
        collection="c",
        source_id=source,
        target_id=target,
        relation_type=kind,
        weight=1.0,
        document_id="doc",
        document_version=1,
        chunk_id=f"chk{n}",
        evidence=f"{source} {kind} {target} ({n}).",
        valid_from=start,
        valid_to=end,
    )


class TestGraphSearch:
    def test_temporal_aggregation(self) -> None:
        items = [
            assertion("a", "b", "supplies", T0, T1, n=1),
            assertion("a", "b", "supplies", T0, None, n=2),
            assertion("c", "b", "supplies", T1, None, n=3),
        ]
        before = aggregate_relations(items, datetime(2024, 6, 1, tzinfo=UTC))
        assert [(r.source_id, r.weight, r.valid_to) for r in before] == [("a", 2.0, None)]
        after = aggregate_relations(items, datetime(2025, 6, 1, tzinfo=UTC))
        assert {(r.source_id, r.weight) for r in after} == {("a", 1.0), ("c", 1.0)}
        closed = aggregate_relations([assertion("a", "b", "x", T0, T1)], T0)
        assert closed[0].valid_to == T1

    def test_link_pagerank_local_search(self) -> None:
        entities = [
            entity("a", "Acme Diagnostics", 5, ("Acme Dx",)),
            entity("b", "Beta Reagents", 3),
            entity("c", "Contoso Freight", 1),
            entity("d", "Dublin"),
        ]
        relations = aggregate_relations(
            [
                assertion("a", "b", "depends_on", T0, None, n=1),
                assertion("b", "c", "supplies", T0, None, n=2),
                assertion("c", "d", "located_in", T0, None, n=3),
            ],
            T2,
        )
        view = GraphView(entities, relations)
        linked = view.link("Who does acme dx rely on? Beta Reagnts maybe")
        assert set(linked) == {"a", "b"}
        assert view.link("") == {}
        ranks = personalized_pagerank(view.adjacency, {"a": 1.0})
        assert sum(ranks.values()) == pytest.approx(1.0)
        assert ranks["b"] > ranks["c"] > ranks["d"]
        assert ranks["a"] > ranks["c"]
        assert personalized_pagerank(view.adjacency, {"zzz": 1.0}) == {}
        result = local_search(view, {"a": 1.0}, max_entities=3)
        assert {e.entity.entity_id for e in result.entities} == {"a", "b", "c"}
        assert [e.entity.entity_id for e in result.entities if e.seed] == ["a"]
        assert {r.relation.relation_type for r in result.relations} == {"depends_on", "supplies"}
        assert result.evidence_chunks[0] == "chk1"
        assert local_search(view, {}) == local_search(view, {})
        assert phrase("depends_on") == "depends on"
        assert phrase("custom_rel") == "custom rel"

    def test_dangling_nodes_keep_mass(self) -> None:
        ranks = personalized_pagerank({"a": {}, "b": {"a": 1.0}}, {"a": 1.0, "b": 1.0})
        assert sum(ranks.values()) == pytest.approx(1.0)

    def test_rank_communities(self) -> None:
        communities = [
            Community(
                community_id=f"com{i}",
                collection="c",
                level=0,
                entity_ids=("a", "b"),
                key_entities=("a",),
                summary=text,
                weight=1.0,
                built_revision=1,
                built_at=T2,
            )
            for i, text in enumerate(
                ["Reagent supply concentration risk", "Logistics in Rotterdam"]
            )
        ]
        ranked = rank_communities("reagent supply risk", communities, {"a": "Acme"})
        assert [r.community.community_id for r in ranked] == ["com0"]
        assert rank_communities("x", [], {}) == []


class TestSummaries:
    async def test_extractive_summary_uses_only_graph_content(self) -> None:
        entities = (
            entity("a", "Acme Diagnostics", 5),
            entity("b", "Beta Reagents", 3),
            entity("c", "Lyon"),
        )
        relations = (
            Relation(
                source_id="a",
                target_id="b",
                relation_type="depends_on",
                weight=2.0,
                evidence_chunks=("k1",),
                evidence=("Acme Diagnostics depends on Beta Reagents.",),
                valid_from=T0,
                valid_to=None,
            ),
            Relation(
                source_id="b",
                target_id="c",
                relation_type="located_in",
                weight=1.0,
                evidence_chunks=("k2",),
                evidence=(
                    "Beta Reagents is based in Lyon.",
                    "Acme Diagnostics depends on Beta Reagents.",
                ),
                valid_from=T0,
                valid_to=None,
            ),
            Relation(
                source_id="a",
                target_id="c",
                relation_type="related_to",
                weight=0.5,
                evidence_chunks=("k3",),
                evidence=("Acme Diagnostics and Lyon.",),
                valid_from=T0,
                valid_to=None,
            ),
        )
        facts = CommunityFacts(entities, relations, DataClassification.INTERNAL)
        summary = await ExtractiveCommunitySummarizer(max_evidence=2).summarize(facts)
        assert summary.startswith(
            "Community of 3 entities centred on Beta Reagents, Acme Diagnostics and Lyon."
        )
        assert "Acme Diagnostics depends on Beta Reagents." in summary
        assert "is mentioned with" not in summary
        assert (
            summary.count("Acme Diagnostics depends on Beta Reagents.") == 2
        )  # statement + one evidence
        lonely = CommunityFacts((entity("z", "Zeta"),), (), DataClassification.INTERNAL)
        assert (await ExtractiveCommunitySummarizer().summarize(lonely)).startswith(
            "Community of 1 entities centred on Zeta."
        )
        empty = CommunityFacts((), (), DataClassification.INTERNAL)
        assert "no named entities" in await ExtractiveCommunitySummarizer().summarize(empty)
