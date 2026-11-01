# Decisions evaluation results

This is a local shadow-evaluation harness, not an actual Decisions or model-quality result. It never reads app configuration or credentials, creates a network client, calls LiteLLM, or contacts a provider. The baseline is the `legacy` passthrough route; it is not an observed set of existing model decisions.

## Current application semantics

The application setting `DECISIONS_MODE` defaults to `off`:

- `off` makes no Decisions call and runs the existing LiteLLM + LangGraph +
  query-kit research path.
- `shadow` makes one bounded provider attempt when configured, records sanitized
  decision telemetry, and discards the route for application behavior. The
  existing research graph remains authoritative, including after timeout,
  provider error, or invalid response fallback.
- `active` uses the validated route. `research` and `legacy` fallback use the
  research graph; `clarify` and `unsupported_local_documents` return deterministic
  no-search responses; `supplied_text` uses a separate LiteLLM transformation
  call without the research search tool. The active default timeout is 5 seconds
  (capped at 10), with default decision concurrency of one.

The active transformation path is not a quality guarantee: it depends on the
configured generation model and fails safely on an unavailable or empty result.
Provider cancellation cannot prove that an already-running synchronous request
stopped remotely or that billing stopped.

## Paired FRAMES comparison

The paired evaluator compares `no_decisions` (`DECISIONS_MODE=off`) with
`with_decisions` (`DECISIONS_MODE=active`) through the same `AgentService`
boundary. Each pair uses the same question, requested model, application prompt
and service settings, query-kit provider configuration, tool budget, and grading
procedure. Pair order is counterbalanced by a deterministic hash of the pair ID.
The active arm receives an injected provider and the OFF arm receives no
provider; the runner does not make a separate advisory call outside the service.

The shared dataset is
[`data/evaluations/frames/frames-test-20.jsonl`](../../data/evaluations/frames/frames-test-20.jsonl):
the first 20 rows, in source order, of the Google FRAMES test split at pinned
revision `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`. It preserves the original
English questions, reference answers, and Wikipedia evidence URLs. The official
dataset card reports Apache-2.0; see the [dataset card](https://huggingface.co/datasets/google/frames-benchmark),
[pinned source TSV](https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv),
and [local provenance](../../data/evaluations/frames/README.md).

This is an English-only, Wikipedia-grounded, nonrepresentative 20-row slice. It
does not establish broad Indic-language research quality. The configured
providers may not retrieve the linked Wikipedia evidence, so provider coverage,
routing, retrieval, and reasoning are confounded. Live retrieval is not frozen
and generation is stochastic; repeated runs can differ.

The evaluator writes a manifest, one record per arm, paired records, and a
summary. Relevant outputs/metrics include route and fallback status, provider
failure, graph status, tool-call count, citation presence, latency, usage when
reported, exact normalized reference match, a bounded normalized
reference-contained proxy, arm equivalence, and human-review flags. Exact match
and reference-contained results are strict automatic proxies only. A non-match,
uncertain result, failed run, or citation-quality judgment requires human review;
arm equivalence is not correctness. Generation usage may be unavailable and
billing evidence may remain `unknown`.

Dry run (validates the pinned input and writes only a manifest; no model,
Decisions, or retrieval call):

```bash
uv run python scripts/evaluate_agent_comparison.py \
  --frames data/evaluations/frames \
  --limit 20 \
  --repetitions 1 \
  --output-dir tmp/decisions-live/comparison-dry-run
```

Bounded paired live run (explicit authorization required; may incur model,
Decisions, and public-provider charges):

```bash
uv run python scripts/evaluate_agent_comparison.py \
  --live \
  --frames data/evaluations/frames \
  --limit 20 \
  --repetitions 1 \
  --model gpt-6-luna \
  --api-key-env OPENAI_API_KEY \
  --output-dir tmp/decisions-live/comparison
```

Credentials must be provided through an untracked `.env` or process-local
environment; use `DECISIONS_API_KEY`/`OPENAI_API_KEY` for Decisions and the
configured LiteLLM/native provider credential for generation. Do not print,
commit, or report secret values. `--live` is the only switch that enables the
paired calls; `--limit 20 --repetitions 1` is the documented bounded run.

### Paired-live result status

The standardized evaluator is configured for the pinned 20-row FRAMES dataset
in source order, but the authorized bounded run executed only one pair:
`frames-test-r0000` (two records, one per arm). Both arms used the same
`gpt-6-luna` generation model, application settings, retrieval/provider
configuration, prompt, tool budget, and grading procedure; only Decisions mode
differed: OFF versus ACTIVE.

The corrected run recorded:

| Arm | Status | Route | Failure | Latency (ms) | Decisions | Proxy score |
| --- | --- | --- | --- | ---: | ---: | --- |
| `no_decisions` | `failed` | `application_graph` | `agent_failed` | 2355.855 | 0 | unavailable |
| `with_decisions` | `failed` | `research` | `agent_failed` | 833.606 | 1 | unavailable |

There was 1 pair, 2 records, 0 completed records, 0 scored arm-equivalence
pairs, and 1 human-review pair. The active Decisions request itself returned
`ok`, resolved `gpt-6-luna`, and reported 185 total tokens (`185` input,
`0` output); this is transport/decision evidence, not answer-quality evidence.
Both application arms produced no answer and no citations. Generation usage
and billing status are unknown. An earlier bounded attempt also had 1 pair and
2 failed records: OFF was `failed`/`application_graph`, while ACTIVE recorded
Decisions `fallback`/`legacy`, with no usage or proxies. The attempts are not
combined into a quality score.

The exact safe process-local launch used for the corrected run was:

```bash
env -u OPENAI_API_KEY -u DECISIONS_API_KEY -u LITELLM_API_KEY \
  -u LITELLM_API_BASE -u LITELLM_MODEL -u LITELLM_CUSTOM_LLM_PROVIDER \
  LITELLM_MODEL=openai/gpt-6-luna LITELLM_CUSTOM_LLM_PROVIDER=openai \
  LITELLM_API_BASE= LITELLM_TEMPERATURE=0.2 LITELLM_MAX_TOKENS=1200 \
  LITELLM_TIMEOUT_SECONDS=30 LITELLM_STREAMING=true \
  DECISIONS_MODEL=gpt-6-luna DECISIONS_MODE=active \
  uv run --env-file .env python scripts/evaluate_agent_comparison.py \
    --live --frames data/evaluations/frames --limit 20 --repetitions 1 \
    --model gpt-6-luna --api-key-env OPENAI_API_KEY \
    --output-dir tmp/decisions-live/compare-corrected
```

This launch loaded dotenv values only into the child process, removed inherited
credential/provider overrides, and did not print or modify `.env`. Do not retry
this bounded comparison merely to obtain a score.

This is an English-only, nonrepresentative 20-case source-order slice, and the
live retrieval evidence is not frozen. The one-pair smoke cannot support broad
Indic-language, full-dataset, quality, or billing claims. The relevant
controlled-fake persistence integration test remains blocked before its test
body by local PostgreSQL password authentication.

The latest local validation passed **168 selected unit tests** (0 failed, 7
deselected) and Ruff across all 22 changed Python files. The PostgreSQL result
above is preserved existing auth-failure evidence and was not rerun. The
completed setup and 20-case dataset must not be described as a completed
20-case comparison: only one pair ran, and both generation arms failed.

### Current generation blocker and next step

Both paired generation arms failed with sanitized `agent_failed`. A separate
diagnostic that used the project key explicitly on the native generation route
returned HTTP `401`. A local mocked wiring diagnostic showed that an explicitly
supplied `litellm_api_key` reaches the native OpenAI client and Authorization
header; it does not prove real-key validity, provider acceptance, or a root
cause, and it does not support a source fix.

The current sanitized environment audit found nonempty project `.env`
generation settings (`LITELLM_API_KEY`, `LITELLM_API_BASE`, and
`LITELLM_MODEL`) plus `OPENAI_API_KEY`. `DECISIONS_API_KEY` is optional because
the supported `OPENAI_API_KEY` alias satisfies it; an existing process
`OPENAI_API_KEY` can override the `.env` value. The configured generation
probe failed with connection refused and no HTTP response. This is transport
evidence only: it does not establish a missing key, invalid key, failed
entitlement, or a local service.

The lateral-route investigation found no practically verified local generation
alternative. The sanctioned next step is owner-side restoration/verification
of the separately managed private generation proxy, followed by one sanitized
generation probe. Do not retry the comparison until that probe succeeds. Keep
the historical native `401`, the separate successful Decisions transport
request, and the bounded failed application pair as distinct evidence; none
supports quality, billing, or full-dataset claims.

## Reproduce

```bash
PHOENIX_ENABLED=false uv run --no-sync python scripts/evaluate_decisions.py \
  --offline \
  --cases tests/fixtures/decisions/requests.jsonl \
  --responses tests/fixtures/decisions/controlled-responses.jsonl \
  --output tmp/decisions-implementation/replay.json \
  --split all \
  --mode shadow
```

Use `--split dev` (normalized in both input data and selection to the dataset value `development`) or `--split heldout`; `development` is also accepted. `--mode off` performs zero controlled-provider invocations, never reads `--responses`, and records the distinct status `off` with legacy for every case. `--mode shadow` requires an explicit response file. Omitting `--output` creates an exclusive microsecond-timestamped file under `tmp/decisions-implementation/`; an explicit `--output` is caller-selected and may overwrite.

## Measured local run

The controlled fixture contains a small, intentionally varied sample: valid route choices, an unknown choice, duplicate answers, and missing responses. It is not a copied full-model corpus and predictions come only from the separate response file. All current request labels are `draft`; genuinely reviewed cases: **0**. Therefore macro-F1, confusion, research-skip, and clarification-miss counts are wiring diagnostics only and must not be presented as model quality.

The evaluator records per-case route, status, fallback reason, source, exact requested model (`gpt-luna-6.1`), unresolved resolved model, rubric version, and evaluator runtime latency. Runtime latency is not model/provider latency. Controlled provider calls are not actual model calls.

Measured locally on 2026-10-31 (IST date; timestamps in output are UTC) after the refinement:

```text
all / shadow:        60 cases, 60 controlled calls, 0 actual model calls,
                     macro-F1 0.218901, potential research skips 1,
                     clarification misses 11
development / off:   20 cases, 0 controlled calls, 0 actual model calls
heldout / shadow:    40 cases, 40 controlled calls, 0 actual model calls,
                     macro-F1 0.250505, potential research skips 0,
                     clarification misses 7
```

The exact commands used were the reproduction command above with `--split all --mode shadow`, then `--split dev --mode off` without `--responses`, and `--split heldout --mode shadow`. All 60 labels are draft labels and 0 cases are genuinely reviewed. These metrics are wiring/arithmetic diagnostics only: canned examples were manually authored to exercise branches, are not statistically independent evidence, and are not model predictions. Genuine model validation remains blocked by unresolved identity/transport/no-charge access and human adjudication.

Measured command/output summary:

```bash
PHOENIX_ENABLED=false uv run --no-sync python scripts/evaluate_decisions.py --offline --cases tests/fixtures/decisions/requests.jsonl --responses tests/fixtures/decisions/controlled-responses.jsonl --output /tmp/decisions-all.json --split all --mode shadow
# {"case_count": 60, "calls": 60}
PHOENIX_ENABLED=false uv run --no-sync python scripts/evaluate_decisions.py --offline --cases tests/fixtures/decisions/requests.jsonl --output /tmp/decisions-development-off.json --split dev --mode off
# {"case_count": 20, "calls": 0}
PHOENIX_ENABLED=false uv run --no-sync python scripts/evaluate_decisions.py --offline --cases tests/fixtures/decisions/requests.jsonl --responses tests/fixtures/decisions/controlled-responses.jsonl --output /tmp/decisions-heldout.json --split heldout --mode shadow
# {"case_count": 40, "calls": 40}
```

The baseline is not implemented as a measured comparator. `legacy` is only the documented passthrough baseline definition; it is not current-model observed behavior. We explicitly defer baseline comparison and any NLP/rule fitting until labels are reviewed. The root plan's human-label and genuine-validation checkboxes remain open.

## Interpretation rules

- Supported macro-F1 classes are exactly `research`, `clarify`, `supplied_text`, `unsupported_local_documents`, and `legacy`; absent-class precision/recall/F1 is zero.
- `potential_research_skips` is HYPOTHETICAL: it counts `requires_research=true` with predicted `supplied_text` or `unsupported_local_documents`, either of which would bypass search under contemplated semantics. `clarify` defers search and `legacy` is unknown, so neither counts. No action happened in this evaluator; this is not observed safety.
- A material clarification miss is a material-clarification label whose prediction is not `clarify`.
- Per-language diagnostics include case count, fallback route-error count, hypothetical research skips, and clarification misses.
- `source=controlled` means a local response fixture validated by the decision contract. It cannot be relabeled as `model` through payload data.
