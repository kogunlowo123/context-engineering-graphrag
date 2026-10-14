# ADR-0005: Deterministic local implementations behind provider interfaces

- Status: Accepted
- Date: 2026-10-14

## Context

The platform can depend on models in six places: embeddings, reranking,
entity extraction, community summaries, answer generation and claim judging.
Tests must pass from a fresh clone with no keys or network access, some
regulated or air-gapped deployments may not call a generative model at all,
and provider outages must not change what the platform claims to have done.

## Decision

Each dependency is an interface with a deterministic local implementation and
provider-backed ones:

| Interface | Local | Provider-backed |
|---|---|---|
| `Embedder` | `HashingEmbedder` (signed feature hashing of stemmed words, bigrams and character trigrams, L2-normalised) | `BedrockTitanEmbedder`, `OpenAICompatibleEmbedder` |
| `CrossEncoder` | `LexicalCrossEncoder` | `HttpCrossEncoder` (`/rerank`) |
| `EntityExtractor` | `PatternEntityExtractor` | `ModelEntityExtractor` (output grounded against the source text) |
| `CommunitySummarizer` | `ExtractiveCommunitySummarizer` | `ModelCommunitySummarizer` (falls back to extractive) |
| `AnswerGenerator` | `ExtractiveAnswerGenerator` | `ModelAnswerGenerator` |
| `EntailmentJudge` | `LexicalEntailmentJudge` | `ModelEntailmentJudge` |

- Generative calls go through the agentic-core `ModelRouter` built from
  `configs/providers.yaml`, with the content guard on input and output.
- Settings select each implementation independently; production refuses the
  hashing embedder; any model-backed feature requires a provider file.
- Provider adapters are tested against recorded wire shapes with
  `httpx.MockTransport` and a fake Bedrock client.

## Consequences

- `uv sync && make check` passes with zero credentials, and results are
  reproducible across machines.
- The extractive generator cannot invent facts; the local mode is a valid
  conservative production choice, not only a test double.
- A collection is bound to the embedding model that built it; changing the
  embedder requires re-ingesting into a new collection, and the service
  refuses mismatched collections.
