# Compliance control mapping

The evidence column names the artifact an auditor can inspect. HITRUST CSF
references are at control-category level.

| Control objective | HIPAA Security Rule | HITRUST CSF | SOC 2 (TSC 2017) | ISO/IEC 27001:2022 | NIST SP 800-53 r5 | NIST AI RMF 1.0 | 21 CFR Part 11 / GxP | Evidence |
|---|---|---|---|---|---|---|---|---|
| Unique, authenticated callers | 164.312(d) | 01 Access Control | CC6.1 | 5.16, 5.17 | IA-2, IA-9 | GOVERN 1.4 | 11.10(d), 11.300 | Bearer verification (`security/authorization.py`), fabric `/v1/authorize`, `tests/security` T1–T3 |
| Least privilege per action and classification | 164.308(a)(4), 164.312(a)(1) | 01 Access Control | CC6.3 | 5.15, 8.3 | AC-3, AC-6 | MANAGE 2.4 | 11.10(g) | `policies/context-governance.yaml`, `authorize:rag:*` audit events, T4, T5 |
| Purpose of use for restricted data | 164.502(b) minimum necessary | 13 Privacy Practices | P4.1 | 5.34 | AC-3(10), PT-2 | MAP 1.1 | — | `deny-restricted-without-purpose`, `X-Purpose-Of-Use` header, T5 |
| Bounded delegation | 164.308(a)(3) | 01 Access Control | CC6.2 | 5.18 | AC-6(9), AU-10 | GOVERN 2.1 | 11.10(g) | `deny-deep-delegation-on-restricted`, `approve-delegated-erasure`, audit `chain`, T13 |
| Content integrity and provenance | 164.312(c)(1) | 10 Information Systems Acquisition | PI1.2 | 5.33, 8.11 | SI-7 | MEASURE 2.9 | 11.10(a), 11.10(c) | Document and chunk SHA-256 digests, versions, citations with `digest`, `GET /v1/collections/{c}/documents/{id}` |
| Untrusted content isolation | 164.308(a)(1)(ii)(B) | 09 Communications and Operations | CC7.1 | 8.7 | SI-3, SI-10 | MANAGE 2.2 | — | Chunk quarantine, `document.ingest` audit `quarantine_reasons`, T6, T7 |
| Secret and credential protection | 164.312(a)(2)(iv) | 06 Configuration Management | CC6.1 | 8.12, 8.24 | SC-12, SC-28 | MANAGE 2.3 | 11.300(b) | Ingestion redaction (T8), KMS, secret manager, External Secrets |
| Output accuracy and human oversight | — | — | PI1.3 | 5.37 | SI-10 | MEASURE 2.5, MANAGE 1.3 | 11.10(f) | Groundedness report in every `/query` response, release rules, `answer.release` audit, T9 |
| Tamper-evident audit trail | 164.312(b) | 09 Communications and Operations | CC7.2 | 8.15 | AU-9, AU-10 | MEASURE 2.8 | 11.10(e) | Signed hash chain (`verify_chain`), WORM bucket, T11 |
| Right to erasure / disposal | 164.310(d)(2)(i) | 13 Privacy Practices | P4.3 | 5.34, 8.10 | MP-6, SI-12 | GOVERN 1.6 | 11.10(c) | `DELETE /v1/collections/{c}/documents/{id}`, `document.erase` audit with counts |
| Data minimisation in logs | 164.502(b) | 13 Privacy Practices | P3.1 | 5.34 | AU-3, PT-3 | MAP 4.1 | — | Audit records `query_digest`/`answer_digest`, not text (`tests/unit/test_platform.py::test_audit_chain_is_signed_and_free_of_query_text`) |
| Change control | 164.308(a)(8) | 06 Configuration Management | CC8.1 | 8.32 | CM-3 | GOVERN 1.4 | 11.10(k) | Git history, CODEOWNERS, `policy_digest` in decisions, contract drift check in CI |
| Model and supplier governance | 164.308(b)(1) | 05 Third Party Assurance | CC9.2 | 5.19, 5.21 | SR-3, SA-9 | GOVERN 6.1 | — | `configs/providers.yaml` classification ceilings, scoped Bedrock IAM in `deployments/terraform/aws` |
| Monitoring and response | 164.308(a)(6) | 11 Incident Management | CC7.3 | 5.24, 8.16 | IR-4, SI-4 | MANAGE 4.1 | — | `configs/prometheus/alerts.yaml`, `docs/runbooks/incident-response.md` |
| Availability and recovery | 164.308(a)(7) | 12 Business Continuity | A1.2 | 8.13, 8.14 | CP-9, CP-10 | — | 11.10(c) | Multi-AZ PostgreSQL with PITR, PDB, `docs/runbooks/disaster-recovery.md` |
| Retrieval quality measurement | — | — | PI1.1 | — | SA-11 | MEASURE 2.3 | — | `POST /v1/collections/{c}/evaluations/retrieval` (recall, precision, MRR, nDCG) |

## GxP and computer software assurance

The platform supports a risk-based CSA approach. Requirements are the
behaviours in `docs/architecture/agent-specification.md`; the automated
tests in `tests/` are the scripted assurance activities. The GitHub Actions
CI run for a release tag (lint, types, SAST, contract drift, unit, security,
chaos, performance, end-to-end and PostgreSQL/pgvector integration suites)
is retained as OQ evidence for that version, together with the release SBOM
and image provenance attestation. IQ is the Helm release record plus the
`/readyz` and Agent Card checks in `docs/runbooks/startup.md`; PQ uses the
retrieval evaluation endpoint against the customer's golden set.

## Right to erasure

`DELETE /v1/collections/{collection}/documents/{document_id}` removes the
document, its chunks, pgvector rows, mentions and assertions, entities that
no longer have mentions, and every community summary of the collection,
because summaries may quote erased text. The response reports the counts and
the same counts are written to the signed `document.erase` audit event.
Other replicas stop serving the content at their next request because the
collection revision changes. See `docs/runbooks/reindex-and-erasure.md`.

## Data minimisation

Audit events for retrieval, context and answers store SHA-256 digests of the
question and answer, chunk identifiers and policy digests, not the text.
Bearer credentials are never stored or logged.
