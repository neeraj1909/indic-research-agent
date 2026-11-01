# ADR: Decisions transport boundary

- **Status:** Implemented prototype; latest bounded evaluator probe succeeded at
  the provider transport boundary, while application integration remains
  unverified and the expected Markdown report is missing.
- **Date:** 2026-10-31
- **Scope:** OpenAI Decisions experiment only; Phase 1 chat generation remains
  on LiteLLM.

## Current decision

Keep the implemented Decisions prototype behind its isolated evaluator and
provider boundary. The installed LiteLLM surface has no Decisions resource or
method, so the prototype uses one direct HTTP request in a standalone CLI for
the bounded live evaluation while preserving LiteLLM for Phase 1 generation.
This direct transport is not application routing: it is not wired into
Chainlit or LangGraph and makes no claim about model quality.

Each evaluator/probe is constrained to the documented `gpt-6-luna` model, one
selected case, one request, and no retry. Controlled/offline tests remain the
source of contract evidence; live results are transport observations. A
successful evaluator response does not establish Chainlit/LangGraph
application integration, model quality, or billing amount.

## Evidence

The current official Decisions guide and create reference document:

- `POST /v1/decisions`, with typed predicate, choice, or score questions.
- `gpt-6-luna` as the only documented model.
- A billed API endpoint; no verified free/no-charge route.

Sources fetched on 2026-10-31:

- [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- [Create decision reference](https://developers.openai.com/api/reference/resources/decisions/methods/create)

The installed stack is LiteLLM 1.83.14, `langchain-litellm` 0.6.4 and OpenAI
SDK 2.24.0. Static inspection found no `litellm.decisions`,
`litellm.adecisions`, `OpenAI.decisions`, or `client.decisions.create` surface.
The prototype therefore uses an isolated direct HTTP CLI rather than claiming
unsupported LiteLLM functionality. LiteLLM remains the generation layer.

The user explicitly selected `gpt-6-luna` and authorized bounded live requests
for Indic. That instruction supersedes the prior blanket assumption that no
live request or additional spend was permitted for this Indic experiment. The
ClauseLens learning-project zero-additional-spend constraint remains unchanged
and is not relaxed by this ADR.

### Latest bounded Indic evaluator result

- Probe status: sanitized JSON reports `ok`; the expected Markdown report is
  missing and is recorded as an evidence blocker.
- Request count: exactly one; no retry or separate connectivity probe.
- Route: `research`.
- Model requested and resolved: `gpt-6-luna`.
- Result: HTTP `200`; provider success validated for the isolated evaluator.
- Usage: 152 total tokens; latency: 1,969.226 ms.
- Chainlit/LangGraph application integration: not exercised or established.
- Billing amount and model quality: not established; do not infer either from
  this single call.

The sanitized JSON is recorded at
`tmp/decisions-live/indic-key-20261031T095211Z-report.json`; the requested
`tmp/decisions-live/indic-key-probe-report.md` was absent when this ADR was
updated.

### Prior bounded live result

- Case: `dev-en-18`.
- Model requested: `gpt-6-luna`.
- Request count: exactly one.
- Retry/connectivity probe: none.
- Endpoint requested: official `/v1/decisions` endpoint.
- Result: HTTP `401` authentication failure (`auth_error`).
- Provider success: no; no route was returned.
- Usage, billing amount, resolved provider identity, and model quality: not
  established for this failed request.

This is an observed authentication failure, but it does not establish that the
model is absent or that the route is unsupported. Credentials were not recorded.
The sanitized result is recorded in `tmp/decisions-live/final-live-probe-report.md`
and `tmp/decisions-live/final-live-result.json`.

### Bounded compatible-endpoint result

- Case: synthetic user-directed probe.
- Model requested: `gpt-6-luna`.
- Request count: exactly one.
- Retry/connectivity probe: none.
- Endpoint requested: private OpenAI-compatible endpoint; its path and identity
  are intentionally omitted from this public ADR.
- Result: sanitized HTTP `4xx` provider error; exact reason/status was not
  observed in the report.
- Route: not recorded; response validation did not succeed.
- Usage, billing amount, resolved provider identity, and model quality: not
  established.

OpenAI-compatible chat support does not establish Decisions support. This
sanitized `4xx` result has an unknown cause and must not be described as proof
that the endpoint is unsupported or that credentials are invalid. The probe
report records no route and no retries. The existing CLI targets the official
`api.openai.com` endpoint, not this private endpoint, and remains an isolated
evaluator rather than graph integration.

## Validation evidence

The implementation and re-review reports record:

- Focused live adapter/CLI unit tests: **15 passed**.
- Relevant decision/evaluator tests: **48 passed**.
- Ruff format/check: passed.
- Compileall: passed.
- `git diff --check`: passed.

These are offline contract and transport-boundary tests. They do not establish
provider success, application routing, or model quality.

## Consequences

- The application retains LiteLLM for Phase 1 generation, `query_cli` and
  BM25-first retrieval; Chainlit remains a thin adapter.
- The isolated evaluator can be rerun with the same one-request/no-retry bound;
  the missing Markdown report should be recovered or regenerated before treating
  the latest evidence as complete.
- The latest successful evaluator request must not be described as Chainlit or
  LangGraph integration, model validation, quality evidence, or billing proof.
- No application routing, enforcement, fallback model, or quality claim is
  authorized by this prototype.

## Remaining blocker

The latest evaluator/provider transport succeeded, but the expected sanitized
Markdown report is missing. Chainlit/LangGraph application integration remains
unverified because the probe was isolated from the application graph. The prior
official endpoint's 401 and compatible endpoint's sanitized 4xx remain
historical results; the latter still has insufficient evidence to determine its
cause. Do not read or expose credentials in reports. If a future request is
authorized, preserve the one-case, one-request, no-retry bound and record a
sanitized outcome. LiteLLM must not be described as Decisions-capable unless
the exact installed/runtime version gains verified support.

## Historical decision and reasoning (superseded for Indic live evaluation)

The following is preserved from the original ADR. Its offline-contract and
LiteLLM compatibility reasoning remains useful; its blanket no-live/no-
additional-spend gate is superseded for this Indic experiment by the explicit
user authorization recorded above.

### Original status

- **Status:** Proposed; live transport blocked. Offline provider seam accepted.

### Original decision

Keep Decisions behind a small injected provider interface that can be exercised
with controlled local responses. Default behavior must remain disabled/offline.
Do not add a live Decisions client, direct SDK call, direct HTTP call, or
`adecisions`/`client.decisions.create` usage until the identity, transport and
cost gates below are resolved.

### Original evidence

The current official Decisions guide and create reference document:

- `POST /v1/decisions`, with typed predicate, choice, or score questions.
- `gpt-6-luna` as the only documented model.
- A billed API endpoint; no verified free/no-charge route.

Sources fetched on 2026-10-31:

- [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- [Create decision reference](https://developers.openai.com/api/reference/resources/decisions/methods/create)

The requested application identity `gpt-luna-6.1` is not proven to be an alias
for official `gpt-6-luna`. The CLI coding-worker identity and transport are a
different account/product boundary from the application's Decisions transport;
a successful `gpt-5.6-luna` worker probe therefore cannot establish the
application model's Decisions entitlement, model mapping, endpoint access, or
pricing. This ADR resolves no identity mapping.

The locked project surface is LiteLLM 1.83.14, `langchain-litellm` 0.6.4 and
OpenAI SDK 2.24.0. Static inspection of the installed packages found no
Decisions resource or Decisions method. Current LiteLLM documentation covers
OpenAI chat/Responses calls and generic proxy pass-through, but does not prove
a typed Decisions SDK route for this locked version. The documentation also
mentions Decisions in newer route/federation notes; that is not a certification
of 1.83.14 and does not authorize an upgrade.

### Original consequences

- Offline contract tests and replay can proceed without credentials, network,
  spending, dependency changes, or model substitution.
- The application retains LiteLLM for Phase 1 generation, `query_cli` and
  BM25-first retrieval; Chainlit remains a thin adapter.
- Live evaluation and enforcement remain blocked. Controlled fixtures must not
  be reported as genuine model evidence.
- A later transport decision must explicitly choose a LiteLLM-supported route
  or amend the LiteLLM constraint before any direct route is considered.

### Original gates before live transport

1. Canonically map `gpt-luna-6.1` to an authorized Decisions model, or obtain
   an explicit user decision to use the documented `gpt-6-luna`.
2. Verify a supported LiteLLM transport against the actual locked/runtime
   version; do not infer support from generic pass-through documentation.
3. Verify endpoint authorization and no-additional-charge access. No live call
   is permitted under the current zero-spending constraint.
4. Validate the Decisions request/response contract with controlled fixtures
   before any real request, including refusal, identity, answer type and value
   validation.
