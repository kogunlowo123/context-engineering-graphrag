# ADR-0001: Hybrid retrieval with reciprocal rank fusion, reranking and MMR

- Status: Accepted
- Date: 2026-10-14

## Context

Enterprise questions mix exact identifiers (supplier names, part numbers,
regulation clauses) that lexical search handles well with paraphrased intent
that dense embeddings handle well. Either retriever alone misses a class of
questions. Their scores live on incompatible scales (unbounded BM25 versus
cosine), so a weighted sum of raw scores needs per-corpus tuning that drifts
as the corpus changes. Top results are also often near-duplicates (several
chunks of the same paragraph), which wastes the context budget.

## Decision

- Run Okapi BM25 (k1 = 1.2, b = 0.75, Lucene idf) over stemmed,
  stop-word-filtered terms and dense cosine search, each to a candidate depth
  (default 50).
- Fuse with reciprocal rank fusion, `sum_i w_i / (k + rank_i)` with k = 60,
  which uses ranks only and needs no score calibration. Ties break by best
  individual rank, then id, so results are deterministic.
- Rerank the fused head (default 30) with a cross-encoder behind the
  `CrossEncoder` interface: a hosted model through the `/rerank` wire format
  in production, a deterministic lexical scorer (coverage, proximity, phrase
  overlap) otherwise. Final score = 0.8 rerank + 0.2 normalised fused score.
- Select the final k with maximal marginal relevance (lambda 0.7) over the
  chunk embeddings to remove near-duplicates.
- Every hit reports its BM25 rank and score, dense rank and score, fused score
  and rerank score, so ranking decisions are explainable.

## Consequences

- No per-corpus score tuning; adding a retriever is one more ranked list.
- A hosted reranker adds one call per query; callers can disable reranking
  per request, and a reranker outage returns 503 instead of silently
  degrading ranking.
- Retrieval quality is measured, not assumed: the retrieval evaluation
  endpoint reports recall, precision, MRR and nDCG on golden sets, so changes
  to any of these parameters can be compared before release.
