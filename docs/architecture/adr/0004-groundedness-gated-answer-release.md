# ADR-0004: Groundedness-gated answer release

- Status: Accepted
- Date: 2026-10-14

## Context

An agent acts on the answer it receives. A fluent answer with plausible
citations can still contain claims its sources do not support, numbers that
differ from the source, or citation labels that point at nothing. Checking
this after the agent has acted is too late, and leaving the check to each
calling agent produces inconsistent controls.

## Decision

- Every answer sentence must carry citation labels that resolve to sources in
  the context pack (`S` passages, `F` graph facts, `C` community summaries).
  Labels that do not exist are stripped and counted.
- Every claim is judged against the sentences of the sources it cites:
  supported, partial, contradicted (numbers or negation disagree) or
  unsupported; a named entity the evidence never mentions caps a claim at
  partial. Groundedness = (supported + 0.5 partial) / claims. The judge is an
  interface: lexical by default, a model judge when configured.
- Release is a policy decision (`answer:release` in
  `policies/context-governance.yaml`): deny contradicted answers, answers
  below 0.5 groundedness and answers with fabricated citations; require
  approval for restricted-source answers below 0.8; allow everything else,
  including abstentions.
- Withheld and pending answers return the plan, citations and groundedness
  report but not the answer text; the audit chain records the answer digest.

## Consequences

- Agents receive one of four dispositions (`answered`, `abstained`,
  `withheld`, `pending_approval`) and can route accordingly.
- Thresholds are reviewed policy data with a digest in every decision, not
  code constants.
- The lexical judge is conservative: paraphrased but correct answers can be
  rated partial. Deployments that generate with a model should judge with a
  model as well, and watch `graphrag_answer_groundedness` and
  `graphrag_answers_total` after changes.
