# ADR-0006: Ingestion content controls and the erasure cascade

- Status: Accepted
- Date: 2026-10-14

## Context

Retrieved documents are the main path for indirect prompt injection: text
written to be read by a model ends up in its context. Documents also carry
credentials pasted into runbooks and tickets, which then flow into
embeddings, prompts and answers. Erasure requests must remove a document
everywhere it was copied: chunks, vectors, graph facts and summaries that
quote it.

## Decision

- **Secret redaction first:** the agentic-core secret scanner runs on the
  submitted text and findings are replaced with `[REDACTED:<kind>]` before
  chunking, so credentials never reach storage, embeddings or models. The
  count is recorded; the value is not.
- **Quarantine per chunk:** each chunk is assessed by the prompt-injection
  detector with a threshold that depends on the declared trust of the source
  (external content is screened most strictly; callers cannot claim SYSTEM
  trust). Quarantined chunks are stored for review but never sent to an
  embedding provider, never indexed and never retrievable; the reasons are
  audited and `graphrag_quarantined_chunks_total` alerts on spikes.
- **Isolation at use:** content that does enter context is rendered between
  random spotlighting boundaries and passed to models as external data.
- **Erasure cascade:** deleting a document removes its chunks, pgvector rows,
  mentions and assertions, entities left without mentions together with their
  assertions and blocking keys, and every community summary of the
  collection, in one transaction that bumps the revision.

## Consequences

- Heuristic detection has false positives; quarantined chunks remain
  inspectable through the document API and can be re-ingested from a cleaned
  source.
- Erasure invalidates all community summaries of the collection; the next
  graph build regenerates them from what remains.
- Audit events keep the erasure counts but not the erased content.
