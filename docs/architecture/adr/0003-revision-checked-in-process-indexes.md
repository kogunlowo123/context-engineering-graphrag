# ADR-0003: Revision-checked in-process indexes, with pgvector as the scale-out path

- Status: Accepted
- Date: 2026-10-14

## Context

Several replicas serve the same collections. Retrieval needs a BM25 index,
dense vectors and a graph view per collection. A dedicated search cluster or
vector database adds a second stateful system to secure, back up, erase from
and keep consistent with PostgreSQL. A message bus to invalidate replica
caches adds another dependency and failure mode.

## Decision

- PostgreSQL is the single source of truth: chunks are stored with their
  float32 embedding and their precomputed index terms.
- Each replica caches, per collection, the BM25 index, the embedding matrix
  and the current graph view together with the collection `revision` they
  reflect. Every write that changes retrievable content increments the
  revision in the same transaction.
- Before serving a request the replica reads the revision (one primary-key
  lookup) and rebuilds under a per-collection lock if it changed. No bus is
  needed, and a replica never serves content older than the last committed
  write it can observe.
- With `GRAPHRAG_VECTOR_BACKEND=pgvector`, vectors are also written to a
  `chunk_vectors` table with an HNSW cosine index and nearest-neighbour search
  runs in PostgreSQL; results are filtered against the replica's view.

## Consequences

- Exact dense search in process is simple and deterministic; its cost grows
  linearly with collection size, and the embedding matrix occupies about
  chunks x dimensions x 4 bytes per replica.
- pgvector removes the linear search cost; replicas still keep the matrix for
  MMR, so very large collections are planned by replica memory.
- A write makes readers rebuild the collection index once; frequent small
  writes to a large collection cause frequent rebuilds, so bulk loads should
  be batched.
- Erasure is a database transaction; no secondary index can retain erased
  content past the next revision check.
