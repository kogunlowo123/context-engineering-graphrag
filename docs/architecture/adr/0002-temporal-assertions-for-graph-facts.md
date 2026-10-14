# ADR-0002: Temporal assertions for knowledge-graph facts

- Status: Accepted
- Date: 2026-10-14

## Context

Enterprise facts change: suppliers are acquired, contracts expire, policies
are superseded. A graph that overwrites edges on re-ingestion answers "who
owns this supplier?" correctly today but cannot answer "who owned it when we
signed the contract?", and loses the evidence behind decisions already taken.
A graph that never retires edges mixes current and obsolete facts in the same
context.

## Decision

- Store relations as assertions: one row per relation stated by one document
  version, with the evidence sentence, source chunk, `valid_from` (the
  version's `effective_at`) and `valid_to` (the document's `expires_at`).
- Ingesting a new version of a document closes that document's open
  assertions at the new version's `effective_at`, in the same transaction
  that writes the new ones. A new version may not take effect before the
  current one.
- Materialise the graph for an instant by aggregating assertions whose
  interval contains it; weights add up across documents and evidence is kept
  per relation. Current views are cached per collection revision; `as_of`
  views are built on demand.
- Apply the same interval to passages: a document is retrievable only between
  `effective_at` and `expires_at`.

## Consequences

- `as_of` questions and neighbourhoods reproduce what was known at the time,
  which supports reviews of past agent decisions.
- History grows with re-ingestion; closed assertions are small (ids, weight,
  one sentence) and erasure removes them with their document (ADR-0006).
- Entities are not versioned; an entity whose mentions disappear remains
  while historical assertions reference it.
