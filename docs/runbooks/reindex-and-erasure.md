# Runbook: re-indexing and erasure

## Changing the embedding model

Each collection records the `embedding_model` and `embedding_dimension` it
was built with. If the service is configured with a different embedder,
every call on that collection returns 409 `conflict` ("Re-ingest into a new
collection"), so vectors from two models are never mixed.

1. Deploy the new embedding settings (`graphrag.embedding.*`) to a staging
   release first.
2. Create a new collection with the same classification and owner
   (`POST /v1/collections`).
3. Re-ingest every document from its source into the new collection.
4. Build the graph: `POST /v1/collections/{new}/graph/build`.
5. Compare retrieval quality on the golden set
   (`POST /v1/collections/{c}/evaluations/retrieval`) for old and new.
6. Point callers at the new collection, then erase the old documents.

Changing chunking settings only affects documents ingested afterwards;
re-ingest a document with a new `chunking` block to re-chunk it (it becomes
a new version).

## Erasure request

1. Confirm the request and identify `collection` and `document_id`
   (`GET /v1/collections/{c}/documents?limit=500`).
2. Erase with a credential holding `rag:document.delete`; delegated agents get
   `approval_required`, so a human operator performs this step:

   ```bash
   curl -X DELETE "$GRAPHRAG/v1/collections/<collection>/documents/<document_id>" \
     -H "Authorization: Bearer $OPERATOR_TOKEN"
   ```

3. Keep the response (counts of chunks, mentions, assertions, entities and
   communities removed) and the matching `document.erase` audit event as
   evidence.
4. Rebuild community summaries, which erasure deletes for the whole
   collection: `POST /v1/collections/{c}/graph/build`.
5. Verify: `GET .../documents/<document_id>` returns 404 and a search for a
   distinctive phrase no longer returns the document on any replica.
6. Backups: PostgreSQL PITR copies age out with the retention period; record
   the date after which no backup contains the document.
