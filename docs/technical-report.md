# indic-research-agent: A Keyword-First, Provider-Backed Research Assistant with Bounded LLM Tool Orchestration

**Technical report draft**
**Prepared:** 29 September 2026
**Repository revision examined:** 6067cca (main)
**Scope:** Current repository behavior, architecture, reproducibility evidence, and a publication-oriented evaluation plan

> This is a source-grounded technical report draft, not a claim of completed
> publication-grade evaluation. All effectiveness claims are explicitly marked
> as measured, unmeasured, or proposed.

## Abstract

Indic-language research spans multiple scripts, transliteration conventions,
code-mixed expressions, low-resource domains, and unevenly indexed scholarly
material. A research assistant for this setting must therefore do more than
generate plausible prose: it must expose how evidence was found, keep the
retrieval boundary inspectable, handle provider failure, and leave enough
telemetry to evaluate the complete search-to-answer process. This report
documents **indic-research-agent**, a Python application that combines a
Chainlit user interface, a LangGraph tool-calling loop, a LiteLLM-compatible
chat-model boundary, and the independently packaged **query-kit** public research
search component. The current runtime is deliberately bounded. The agent can
issue typed public-provider searches, collect structured results, synthesize an
answer, stream progress, persist selected audit and conversation data, and cache
repeated calls. It does not currently ingest a local corpus, run a local BM25
index, use embeddings or vector databases, or establish retrieval or answer
quality through a benchmark.

The report uses repository source, tests, recent commit history, public official
documentation, primary research literature, and a public Google Scholar profile
supplied for structural context. It reconstructs the actual system at revision
6067cca, records a documentation-drift hazard in the older architecture
document, and defines an evaluation matrix for future work. Local validation
provides evidence of wiring and deterministic behavior: the independent
query-kit component passes 81 tests and returns three live Semantic Scholar
records for a Hindi OCR query; the application passes formatting/lint checks and
86 unit tests; all four integration tests pass with PostgreSQL and Redis; and
the three selected e2e tests pass when the application is run on an alternate
host port. Consequently, the central research conclusion is deliberately
narrow: the project currently demonstrates a testable, observable
research-assistant scaffold and a live provider boundary; its usefulness,
retrieval effectiveness, language coverage, citation faithfulness, latency,
cost, and user value remain empirical questions.

**Keywords:** Indic NLP, research assistant, tool-using language model,
keyword retrieval, BM25 baseline, LangGraph, LiteLLM, Chainlit, query-kit,
retrieval evaluation, reproducibility

## 1. Introduction

### 1.1 Motivation

Research discovery for Indian and Indic-language topics is unusually sensitive
to query formulation. The same topic may appear under an English name, a native
script, a transliteration, a code-mixed spelling, an acronym, or a dataset name.
Language families, scripts, domains, publication venues, and data availability
vary substantially. An assistant that only produces a fluent summary can hide
whether it found relevant evidence, whether the provider covered the intended
language, and whether the final answer is grounded in the retrieved sources.

The project addresses this problem as an evidence-retrieval and orchestration
problem rather than as a generic chatbot. Its system prompt narrows the domain
to Indic-language and India-focused research, asks for clarification when
language, script, task, domain, timeframe, or provider coverage would materially
change the answer, and directs factual or literature questions through a search
tool. The prompt also imposes a citation contract: retrieved results should be
identified by stable citation labels and the answer should finish with a source
mapping. These are design intentions encoded in
[agent/prompts.py](../src/indic_research_agent/agent/prompts.py); they are not
by themselves measurements of citation correctness.

The current architecture also reflects a practical sequencing decision. The
repository keeps retrieval keyword-first and avoids embeddings and vector
databases. The application calls public research providers through
[services/querykit_service.py](../src/indic_research_agent/services/querykit_service.py)
and exposes a typed search tool rather than pretending that a local, fully
indexed corpus already exists. This is a defensible first boundary for a system
whose data coverage, corpus license, and language-specific benchmark have not
yet been established. It is also a falsifiable baseline: later work can measure
where keyword retrieval succeeds or fails before adding a more complex
retrieval layer.

### 1.2 Research questions

This report answers four engineering and research questions:

1. **What is implemented now?** Which modules, boundaries, data contracts, and
   runtime behaviors exist at revision 6067cca?
2. **Why is the architecture shaped this way?** How does the design relate to
   sparse retrieval, retrieval-augmented generation, tool-using agents, and
   Indic NLP constraints?
3. **What has been demonstrated?** Which claims are supported by source review,
   unit tests, a live provider-component check, or an end-to-end run, and which
   are still hypotheses?
4. **What should be measured next?** What benchmark, ablation, quality, safety,
   latency, cost, and user-centered evidence would turn the scaffold into a
   research-grade evaluation?

### 1.3 Contributions of this report

The report contributes a current-state reconstruction rather than a new
retrieval algorithm. Its concrete contributions are:

- a module-level description of the running application and its provider
  boundary;
- an explicit distinction between the current public-provider path and stale
  local-BM25/FetchTool descriptions in docs/architecture.md;
- a source-grounded discussion connecting the implementation to BM25, ReAct,
  RAG, agent evaluation, and Indic-language research;
- a reproducibility record for the tests and isolated query-kit check that
  were executed;
- an evaluation protocol that can measure retrieval, tool use, grounded
  generation, reliability, safety, cost, and Indic-language coverage without
  prematurely claiming success.

The report does not claim that the application is a new language model, a new
retrieval algorithm, a production-scale search engine, or a validated solution
for all Indic languages.

## 2. Background and related work

### 2.1 Sparse retrieval and the BM25 baseline

BM25 is a probabilistic, term-based ranking family that adjusts the contribution
of term frequency by document length and tunable saturation/normalization
parameters. Robertson and Zaragoza's review provides the standard scholarly
background [1], while the Stanford Information Retrieval book gives an
implementation-oriented explanation of the Okapi BM25 formulation [2]. The
important engineering implication is not that BM25 is universally optimal. It is
that a sparse baseline is inspectable, inexpensive, and strong enough to expose
whether a proposed semantic layer is solving a measured problem.

The present project must be described carefully. Its architecture contract is
keyword/BM25-first, but the current application no longer contains a local
document index or a retrieval package. The live runtime sends typed queries
to public providers through query-kit; provider-side ranking is therefore an
external implementation detail. The repository's historical
[docs/architecture.md](architecture.md) still describes local BM25 indexing,
seed documents, and a FetchTool, but the recent commits b4d233a, 517ba14,
and f0d9e77, the current source tree, the README, and the unit guard
test_no_local_bm25_mode.py describe a different system. This report treats
the historical document as documentation-drift evidence, not as current runtime
evidence.

The correct research claim is consequently: *the project adopts a
keyword-first/provider-backed retrieval posture and preserves a simple sparse
retrieval baseline as an architectural constraint*. It cannot yet claim a
measured BM25 score, provider ranking advantage, or superiority over dense
retrieval.

### 2.2 Retrieval-augmented generation and grounded answers

Retrieval-augmented generation (RAG) combines a generator with an external
memory or retriever so that answers can use information not contained in model
parameters [3]. The RAG literature is relevant because indic-research-agent
also follows a retrieve-then-synthesize shape. The analogy has limits. The
current project does not build the classic dense-vector RAG stack: there is no
embedding API call, vector database, local document ingestion path, or
pgvector/FAISS/Chroma/Milvus/Weaviate/Pinecone dependency. The agent receives
structured public-provider results as tool messages, then asks the model to
synthesize from the conversation.

This distinction matters for both architecture and evaluation. A RAG benchmark
that assumes a fixed local corpus cannot be copied directly onto a dynamic
provider-backed system. The evaluation must capture provider, query, timestamp,
result identifiers, ranking, failures, and the final answer. Retrieval quality
and generation quality should be scored separately, as recommended by recent
RAG-evaluation work [4]. RAGAS [5] is useful as a vocabulary for faithfulness,
answer relevance, and context relevance, but an automated evaluator should not
be treated as ground truth without human alignment and language-specific
validation.

### 2.3 Tool-using language models and bounded action loops

ReAct frames language-model behavior as an interaction between reasoning traces
and task actions, where actions can gather external information before the model
continues [6]. The project's graph is a bounded engineering variant of this
pattern. It has an llm node, a tools node, conditional routing, and a fixed
default maximum of six tool calls. The tool is typed search; it is not arbitrary
code execution. The graph records custom events, tool metadata, prompt
fingerprints, and trace attributes, which makes the action sequence observable.

The bounded design is intentional. It reduces runaway provider calls and gives
the system a predictable termination condition. When the tool budget is
exhausted, the graph disables tools and injects a final-synthesis instruction;
if a model still returns tool calls, the implementation converts the response to
an answer or a limitation fallback. This is useful operational behavior, but it
does not establish that the model selected the right number of searches or
formed good queries. Agent evaluation surveys emphasize that final-task success
must be complemented by process measures such as tool selection, parameter
correctness, safety, latency, and cost [7, 8].

### 2.4 Indic NLP and low-resource research context

Indian-language NLP has a broad resource and tooling landscape with uneven
coverage across languages, scripts, tasks, and data regimes [9]. IndicBERT
demonstrates the value of multilingual pretraining for a set of Indian
languages, while also illustrating that language coverage is a concrete model
and data choice rather than a generic “Indic” property [10, 18]. A recent review of
parallel corpora for low-resource Indic translation further highlights language
family diversity, script variation, data scarcity, informal/noisy text, and
dialectal variation [11]. These observations motivate evaluation strata for the
assistant: at minimum, language, script, transliteration, code-mixing, domain,
query length, publication time, and provider coverage.

The public Scholar profile supplied with the task lists interests in neural
networks, natural language processing, and explainable AI, and includes public
work on Indic-language hate-speech mitigation. The related SafeSpeech article
uses a conventional applied-paper structure—motivation/objectives/contributions,
related work, methodology, experiments, error analysis, and conclusion/future
work [12]. That structure informed this report's organization. The profile was
used only as public context and a writing-structure reference; it is not evidence
of project ownership, private account data, or project performance.

## 3. Methodology: how the system and evidence were examined

### 3.1 Repository snapshot

The application was examined from a clean repository state at commit 6067cca,
whose recent history includes the transition away from local search and document
storage and the current Chainlit progress-status behavior. The declared project
runtime is Python >=3.12,<3.13. The dependency surface includes Chainlit,
LangChain/LangGraph, LiteLLM, query-kit from GitHub, PostgreSQL/SQLAlchemy,
Redis, and Phoenix instrumentation. The report does not treat version ranges in
pyproject.toml as a frozen experiment environment; a publication release
should record an exact lockfile, image digest, model identifier, provider
configuration, and data snapshot.

The analysis used the following evidence order:

1. current source and tests for implemented behavior;
2. git history and README for architectural intent and removed features;
3. direct component execution for the independent query-kit boundary;
4. official technical documentation for framework API terminology;
5. primary research papers and surveys for related work and evaluation design;
6. the public Scholar page for contextual and structural reference only.

This order prevents a generic framework description or an old local document
from silently becoming a claim about current behavior.

### 3.2 Source and literature review

The literature workflow used headed browser evidence and reviewed rendered pages
rather than treating search snippets as citations. Three query families covered
retrieval/agent foundations, Indic NLP, and agent/RAG evaluation. Selected
sources include the BM25 chapter/review, ReAct, RAG, RAGAS, agent-evaluation
surveys, the Indian-language NLP survey, IndicBERT, the LoResMT review, and
SafeSpeech. Official documentation covered LangGraph streaming, LiteLLM
streaming, Chainlit message/step lifecycles, and service-layer/dependency
injection patterns.

One route was deliberately rejected: a Meta AI RAG landing page returned a large
HTML response but failed the rendered visible-content quality gate twice. The
directly extractable arXiv version was used instead. This is a small but
important reproducibility rule: if a source cannot be inspected as rendered
content, it should not silently become primary evidence merely because its URL
looks authoritative.

### 3.3 Claims policy

The report distinguishes four claim classes:

| Class | Meaning | Example in this report |
|---|---|---|
| Implemented | Directly visible in current source/configuration. | The graph has llm and tools nodes and a six-call default budget. |
| Executed | Observed in a command or test run. | The independent query-kit suite passed 81 tests. |
| Literature-grounded | Supported by a cited paper or official documentation. | Retrieval and agent evaluation should be separated into process and outcome measures. |
| Proposed | A future experiment or design recommendation. | Stratify a benchmark by language, script, and code-mixing. |

No retrieval precision, recall, MRR, NDCG, answer faithfulness score, user
satisfaction score, latency percentile, or cost number is presented as measured
because no such benchmark run exists in the examined repository state.

## 4. Current system architecture

### 4.1 End-to-end flow

The current runtime can be summarized as follows:

~~~text
User message
    |
    v
Chainlit adapter (session, auth, history, one ephemeral status step)
    |
    v
ChatController (thin UI/application boundary)
    |
    v
AgentService.stream_answer / answer
    |
    v
LangGraph: START -> llm -> [tools -> llm]* -> END
    |                  |
    |                  +--> typed SearchTool
    |                                |
    |                                v
    |                         QueryKitService -> query-kit -> public providers
    |
    +--> LiteLLM-compatible model for query planning and synthesis
    |
    +--> streamed events, Phoenix traces, audit persistence, Redis cache
~~~

The flow is an adapter-oriented composition. Chainlit receives the message and
renders events; it does not contain the search policy or the graph's
orchestration. ChatController owns the conversation-facing boundary and
history update. AgentService coordinates settings, history, streaming, cache,
persistence, tracing, and the compiled graph. The graph calls a single typed
search tool. The query-kit service normalizes provider results into the
application result model. Persistence and caching are supporting services rather
than retrieval algorithms.

### 4.2 Module map

| Layer | Current responsibility | Evidence |
|---|---|---|
| UI | Chainlit callbacks, authentication hook, session/thread state, message streaming, ephemeral progress rendering. | [ui/chainlit_app.py](../src/indic_research_agent/ui/chainlit_app.py), [ui/chainlit_stream_renderer.py](../src/indic_research_agent/ui/chainlit_stream_renderer.py) |
| Controller | Normalize history, expose handle_message/stream_message, append completed exchanges. | [controllers/chat_controller.py](../src/indic_research_agent/controllers/chat_controller.py) |
| Agent service | Run or stream a request, connect the graph to cache/persistence/tracing, and return an answer contract. | [services/agent_service.py](../src/indic_research_agent/services/agent_service.py) |
| Graph | Add system prompt, bind typed search, alternate LLM/tool nodes, emit events, trace calls, enforce budget, finalize answer. | [agent/graph.py](../src/indic_research_agent/agent/graph.py), [agent/state.py](../src/indic_research_agent/agent/state.py) |
| Prompt | Define scope, clarification policy, search policy, citation discipline, and runtime boundaries; fingerprint prompt version/hash. | [agent/prompts.py](../src/indic_research_agent/agent/prompts.py) |
| Tool | Validate query, top-k, providers, and year; execute search; return normalized result objects. | [tools/schemas.py](../src/indic_research_agent/tools/schemas.py), [tools/search.py](../src/indic_research_agent/tools/search.py) |
| Provider boundary | Call query-kit, normalize provider aliases/results, preserve source/citation metadata, and provide fallback query variants. | [services/querykit_service.py](../src/indic_research_agent/services/querykit_service.py) |
| Persistence | Store chat/audit/query/provider/cache metadata through SQLAlchemy/Alembic repositories when configured. | [services/chainlit_data_layer.py](../src/indic_research_agent/services/chainlit_data_layer.py), [db/](../src/indic_research_agent/db/) |
| Cache | Use canonical JSON payloads and SHA-256-derived Redis keys with TTL and optional metadata recording. | [services/cache_service.py](../src/indic_research_agent/services/cache_service.py) |

There is no current src/indic_research_agent/retrieval/ directory. There is
also no local document-ingestion runtime in the source tree. The system prompt
explicitly tells the model that local document search is not implemented. This
negative evidence is as important as the positive module map because it bounds
what the application can honestly promise.

### 4.3 Agent state and graph semantics

The state contract contains messages, an optional query identifier, a tool-call
count, retrieved-context payloads, and a final answer. At graph construction,
the search tool is converted into a LangChain structured tool and bound to the
model when the model supports bind_tools.

The llm node performs these actions:

1. count prior tool calls and decide whether the configured budget is exhausted;
2. replace any prior system message with the current system prompt;
3. add a final-synthesis instruction when tools are disabled;
4. invoke the model through ainvoke or a thread-backed synchronous fallback;
5. trace model inputs/outputs and emit start/end/finalized events;
6. store a final answer when no tool call is requested, including a deterministic
   limitation fallback for empty responses.

The tools node iterates over model-requested calls, validates search arguments
with Pydantic, runs the search tool, stores serialized result payloads in
retrieved_context, creates ToolMessage values, traces metadata, and increments
the count. Conditional routing returns to the model while calls remain or ends
when the model has produced an answer or the budget is reached.

This is a small state machine rather than an open-ended autonomous agent. Its
main research value is observability: the system can later measure which query
was selected, which provider was requested, how many results returned, how long
each boundary took, and whether the answer cited the retrieved identifiers.

### 4.4 Search contract and provider boundary

SearchToolInput rejects unknown fields and constrains query to a non-empty
string, top_k to 1–20, and since_year to a positive integer. The normalized
result includes document/chunk identifiers, score, title, source, snippet,
optional citation ID, and metadata. These types give tests and downstream
rendering a stable contract even when individual public providers differ.

The provider service also contains a practical recovery policy. If a long or
narrow query fails to produce usable evidence, known Indic/OCR terms can produce
short provider-friendly alternatives such as Hindi OCR, Devanagari OCR, or
Indic OCR. The policy is useful for robustness but can alter recall and query
intent. It must therefore be logged and evaluated rather than treated as an
invisible implementation detail.

An isolated run of the external component demonstrated the boundary:

~~~text
cd /home/neeraj/Code/query-kit
timeout 60 uv run pytest
# 81 passed in 0.69s

timeout 60 uv run query-cli search "Hindi OCR" --provider semantic-scholar --limit 3 --format json
# succeeded; three records were returned
~~~

The records included recent Hindi OCR work. This is evidence that the component
and provider contract were live in the captured environment. It is not evidence
that the application always selects good queries, that provider rankings are
correct, or that the final LLM answer is faithful.

### 4.5 Streaming, UI, and observability

The Chainlit adapter creates an empty response message, then passes service
events to ChainlitStreamRenderer. Tokens stream into the response when
available. Agent/tool progress updates a single Chainlit step whose metadata
marks it as ephemeral; completion or failure removes that status step. This
implements a thin presentation boundary: the renderer maps framework-neutral
events to Chainlit objects, while the agent service and graph remain testable
without a browser.

This design aligns with the documented responsibilities of the underlying
frameworks. LangGraph exposes stream modes for state updates, messages/tokens,
and custom data [13]. LiteLLM supports streaming model responses through its
completion boundary [14]. Chainlit messages and steps provide send/update and
lifecycle-oriented presentation primitives. The repository's own tests remain
the authority for the specific event mapping and ephemeral-row behavior.

Tracing records LLM and tool spans with model, prompt-version/hash, tool-call
count, provider/result metadata, and latency attributes. The prompt fingerprint
is intentionally a short non-secret hash, allowing a run to be associated with
the prompt version without logging the full prompt as an identity token. A
publication run should additionally record model/provider versions, sampling
parameters, environment, and query corpus snapshot in a reproducible manifest.

### 4.6 Persistence, caching, and deployment context

The application is packaged with Docker Compose services for the app,
PostgreSQL, and Redis. PostgreSQL supports Chainlit history and application
audit/query metadata through migrations and repositories. Redis stores canonical
JSON values under namespace plus SHA-256 payload keys with a configurable TTL.
These services improve continuity and operational cost, but they do not change
the retrieval semantics or prove multi-replica production safety.

The persistence boundary is deliberately optional in unit tests. Integration
tests exercise the actual database/cache contract, which is why they failed
when the required services were not running. This is preferable to silently
replacing a live integration dependency with an in-memory fake while reporting
the test as an end-to-end result.

## 5. Implementation details and design rationale

### 5.1 Thin adapters and explicit seams

The separation between Chainlit, ChatController, AgentService, and the graph
follows a service-layer principle: orchestration should be separated from
delivery mechanisms, and explicit dependencies make tests and alternate
composition easier [15, 16]. The practical benefit is that renderer tests can
send synthetic agent events, graph tests can use fake models/tools, and provider
tests can run without a browser. The risk is an anemic service layer if domain
decisions are pushed into generic orchestration functions. The current project
keeps the domain small; future additions should preserve named contracts rather
than grow an unstructured coordinator.

### 5.2 Bounded tool use

The six-call default is an operational guardrail. Each call can incur provider
latency and model cost, and an unbounded loop would make failure diagnosis harder.
However, one fixed budget cannot be assumed optimal across a simple source lookup,
a multi-language comparison, and a literature review. A future experiment
should compare budgets such as 1, 2, 4, 6, and 8 under a fixed task set while
measuring answer quality, source coverage, latency, and cost. The experiment
should also log whether a larger budget produced useful additional evidence or
merely repeated variants.

### 5.3 Provider-friendly query fallback

The search service's fallback variants are an example of targeted robustness
engineering. They address a known failure mode: a model can produce a precise
query that is semantically sensible to a human but too narrow for a public
provider. Short alternatives can recover results. The trade-off is that fallback
queries may broaden the intent or introduce a language/script mismatch. Every
fallback should therefore be recorded with an original-query identifier and
evaluated on both result availability and relevance.

### 5.4 Citation discipline as a contract

The prompt requires citation labels, source URLs where available, and a Sources
footer. The typed result model carries citation_id and metadata to make that
possible. This is a good interface design because citation construction is
possible without parsing arbitrary prose from a provider. It is not sufficient
for citation correctness. The evaluator must check whether each cited source
actually supports the associated claim, whether a source was retrieved or merely
remembered, and whether the answer omits important contradictory evidence.

### 5.5 Security and privacy posture for a research assistant

The application stores query and audit information and integrates authentication,
browser/session state, tracing, and external providers. A production evaluation
must therefore inspect prompt-injection resistance, provider data handling,
secret redaction, user/session isolation, and retention policy. The current
technical report intentionally does not reproduce credentials or account-only
Scholar content. The same principle should apply to trace payloads: a trace
should expose enough source metadata to debug evidence use without retaining
unbounded private conversation data by default.

## 6. Validation and evaluation

### 6.1 Executed repository gates

The following checks were run against the examined application revision:

| Gate | Command | Outcome | What it demonstrates |
|---|---|---|---|
| Format | uv run ruff format --check . | Pass: 86 files already formatted. | Source formatting is stable. |
| Lint | uv run ruff check . | Pass: all checks passed. | Selected static checks pass. |
| Unit | timeout 60 uv run pytest -m unit --no-cov | Pass: 86 passed, 7 deselected; one existing Pydantic deprecation warning. | Fast contracts for graph, tools, services, cache, events, renderer, and constraints pass. |
| Component | cd /home/neeraj/Code/query-kit && timeout 60 uv run pytest | Pass: 81 passed in 0.69 s. | The independent query-kit component passes its suite. |
| Live provider boundary | query-cli search "Hindi OCR" --provider semantic-scholar --limit 3 --format json | Pass: three records returned. | A live public-provider call worked in the captured environment. |
| Integration | timeout 90 uv run pytest -m integration --no-cov | Pass: 4 passed, 89 deselected, 2 existing dependency warnings. | Database schema, repositories, UI persistence, and Redis cache pass with live services. |
| E2E/application | APP_DATABASE_URL=... REDIS_URL=... CHAINLIT_BASE_URL=http://localhost:8001 timeout 90 uv run pytest -m e2e --no-cov | Pass: 3 passed, 90 deselected, 1 existing Pydantic warning. | Auth/persistence, deterministic agent smoke, and streaming contracts pass against the service-backed application. |

The first integration attempt was blocked because no services were running. The
follow-up pass started PostgreSQL and Redis through Compose, ran the integration
suite, ran the deterministic smoke query, and exercised the Chainlit app on
host port 8001. The default host port 8000 was already occupied by an unrelated
running container, so the application container was validated on the alternate
port rather than disturbing that service. This establishes deployment wiring
for the captured environment, not production-scale reliability.

### 6.2 Proposed benchmark design

The current tests establish wiring and invariants. They do not answer whether a
researcher receives better evidence or a better answer. A benchmark should be
constructed from real research tasks, with a held-out evaluation set and source
annotations. Each task should contain:

- the user question and intended information need;
- language, script, transliteration, code-mixing, domain, and recency tags;
- acceptable source IDs or a graded relevance set;
- expected answer claims, when feasible;
- required citation support for each claim;
- provider availability and capture timestamp;
- difficulty label: direct lookup, comparison, synthesis, or multi-hop review.

The first dataset should be small and auditable rather than broad and weakly
annotated. A reasonable pilot could contain 50–100 queries across Hindi,
Marathi, Bengali, Tamil, Telugu, and English/Indic mixed queries, but this is a
proposed sampling target, not an existing project result. Each language should
include native-script and romanized forms where users actually employ both.

### 6.3 Retrieval metrics

For a fixed provider snapshot or replayable result set, measure:

- **Precision@k:** proportion of the top-k results judged relevant;
- **Recall@k:** proportion of known relevant sources recovered;
- **MRR:** rank of the first relevant source;
- **NDCG@k:** graded ranking quality when relevance has more than two levels;
- **coverage:** fraction of tasks with at least one usable result;
- **metadata completeness:** fraction of results with title, URL/source,
  authors/year, and stable citation metadata;
- **provider agreement:** overlap and disagreement across selected providers.

The RAG-evaluation literature warns that keyword matching can miss semantic
matches and that retrieval and generation should be reported separately [4].
For this project, the relevant baseline comparison is not “BM25 versus a
vector database” by assumption. It is: original model query versus normalized
keyword query versus provider-friendly fallback, optionally across providers,
with the exact query and result list frozen. Only after that baseline is measured
should a later plan authorize a semantic retrieval experiment.

### 6.4 Agent-process metrics

For every run, record a replayable event trace containing model identifier,
prompt fingerprint, user query hash or approved raw query, tool-call sequence,
arguments, provider, result IDs, fallback status, failures, and final answer.
Then measure:

- tool-call selection accuracy: did the model search when search was required?
- parameter accuracy: language/script/domain/provider/year/top-k fields;
- query usefulness: relevance of each generated query and marginal result gain;
- redundant-call rate and budget exhaustion rate;
- provider-failure recovery rate;
- final-answer completion rate and limitation-message correctness;
- citation precision: cited sources support the claims they follow;
- citation recall: important answer claims have supporting sources;
- unsupported-claim rate and contradiction rate.

This follows the broader agent-evaluation view that interaction mode, task
outcome, tool behavior, safety, and process instrumentation are all part of the
evaluation surface [7, 8]. A final answer judged in isolation would miss a
system that happens to sound good while searching the wrong provider or citing
irrelevant results.

### 6.5 Generation, latency, cost, and user measures

Generation should be scored with a combination of reference-based checks,
source-supported human judgments, and carefully validated automated metrics.
Useful dimensions are factual correctness, source entailment, completeness,
relevance, clarity, language/script fidelity, and uncertainty calibration. An
LLM judge can assist triage but should be calibrated against human annotations,
especially for code-mixed or native-script answers.

Operational measurements should include time to first token, time to first
source, end-to-end latency, provider latency, model latency, failure rate,
retry/fallback counts, token usage, and estimated cost. Report median and tail
percentiles by query class rather than one average. User evaluation should test
whether researchers can find and verify sources faster, distinguish evidence
from synthesis, and recover from an empty or failed provider response.

### 6.6 Evaluation matrix

| Dimension | Minimum evidence | Current state | Next experiment |
|---|---|---|---|
| Retrieval relevance | Human-labeled relevance set plus Precision/Recall/MRR/NDCG. | Missing. | Build a stratified Indic query set and freeze provider captures. |
| Tool correctness | Gold decision/query parameters and event traces. | Instrumentation exists; gold labels missing. | Annotate when search is required and what constraints matter. |
| Grounding | Claim-to-source support labels. | Prompt contract exists; score missing. | Annotate claims and evaluate citation precision/recall. |
| Robustness | Controlled provider timeout/empty/malformed-result tests. | Unit seams exist; live fault matrix missing. | Replay injected failures and compare fallback behavior. |
| Latency/cost | Timestamped spans and model/provider billing data. | Some tracing fields exist; benchmark absent. | Run repeated queries under fixed model/configuration. |
| Indic coverage | Per-language/script/code-mix breakdown. | One live Hindi OCR probe only. | Add multilingual and transliterated tasks. |
| User value | Task completion/time-to-source and qualitative review. | Missing. | Conduct a small researcher study with ethics/privacy review. |

## 7. Discussion

### 7.1 What the current design gets right

The strongest aspect of the current implementation is boundary clarity. The
project does not claim local document search when that feature is absent. It
exposes one typed search tool, keeps the UI adapter thin, and records enough
events to make the path inspectable. This is a more credible starting point for
research than a broad feature list unsupported by executable contracts.

The second strength is isolation. query-kit can be tested and invoked outside
Docker and outside Chainlit. This makes provider bugs distinguishable from
application-wrapper bugs. The app's unit tests can exercise tool schemas,
fallbacks, graph routing, event translation, cache-key behavior, and the
ephemeral status renderer without requiring a live LLM.

The third strength is epistemic signaling in the prompt and fallback behavior.
The agent is instructed to cite retrieved evidence, report thin or unavailable
evidence, and finish with an explicit no-sources limitation when necessary. The
graph also avoids a silent tool-loop failure after budget exhaustion by forcing
final synthesis. These mechanisms do not guarantee good answers, but they make
failure modes more legible.

### 7.2 What remains unresolved

The central unresolved question is retrieval effectiveness for actual Indic
research tasks. Public providers differ in indexing, metadata, query syntax,
rate limits, and language coverage. A live Hindi OCR result proves availability,
not broad coverage. The application also depends on an external LLM for query
formation and synthesis, so provider and model variation can dominate observed
quality.

The second unresolved question is citation faithfulness. The prompt has a strong
contract, but no evaluator yet checks whether each sentence is supported by the
source it cites. In a research assistant, that is a more important target than
fluency alone.

The third unresolved question is whether six calls is a good budget. A small
budget may truncate multi-step literature discovery; a larger budget may waste
cost on redundant searches. The correct value is task-dependent and should be
measured.

The fourth is documentation consistency. The stale architecture page can cause
future contributors or agents to implement against removed local-search
assumptions. The report resolves the ambiguity for this snapshot, but the
repository should eventually update or label that document.

### 7.3 Keyword-first trade-offs

Keyword-first retrieval has practical advantages: query and result behavior are
inspectable, the component is cheap to run, and no embedding model or vector
index has to be selected, hosted, refreshed, or licensed. It can also preserve
literal matches for dataset names, scripts, transliterations, author names, and
technical acronyms.

Its weaknesses are equally concrete. Vocabulary mismatch, spelling variation,
transliteration, morphology, script conversion, and concept-level similarity can
lead to missed evidence. Provider ranking may also be opaque. The appropriate
engineering response is measurement and targeted normalization, not an automatic
claim that keyword search is sufficient or an automatic jump to embeddings.
The current project is well positioned for that comparison if it records exact
queries, fallback paths, providers, and result lists before adding another
retrieval mechanism.

## 8. Threats to validity and limitations

### 8.1 Construct validity

Passing unit tests measures implementation contracts, not research usefulness.
The live provider query measures reachability and output shape, not relevance or
recall. A fluent model answer measures neither grounded correctness nor user
value. Future claims must map each intended construct—retrieval quality,
grounding, robustness, or productivity—to an explicit metric and annotation
protocol.

### 8.2 Internal validity

Model sampling, provider updates, rate limits, cache state, prompt version,
fallback behavior, and network failures can all change outcomes. Evaluation
runs need a fixed model/configuration, prompt fingerprint, query capture,
provider/timestamp metadata, cache policy, and replay or snapshot strategy.
Repeated runs are necessary for stochastic model behavior.

### 8.3 External validity

One Hindi OCR query cannot represent Indic-language research. Results from one
provider or one model cannot be generalized to all public scholarly search. A
benchmark should stratify by language, script, domain, query type, and recency,
and should report missing or unavailable provider coverage rather than silently
dropping those tasks.

### 8.4 Reproducibility and dependency drift

The repository declares dependency ranges and a Git dependency for query-kit.
The public provider and model services are external and mutable. Reproduction
requires exact package resolution, commit SHAs, container image digests,
environment configuration, model/provider identifiers, and captured result
metadata. The integration gate also depends on PostgreSQL and Redis being
available. The current report records the missing-service failure explicitly so
that a future run can distinguish environmental repair from application change.

### 8.5 Privacy, safety, and ethics

Research queries may contain unpublished topics, sensitive social categories,
personal data, or legally sensitive material. External provider and LLM calls
may create retention or jurisdiction concerns. Tool traces can expose source
queries and snippets. Before a public deployment or user study, the project
needs a data-minimization policy, secret/PII redaction tests, provider terms
review, prompt-injection tests, access-control verification, and an approved
retention period. The supplied Scholar profile was handled under a public-only
boundary; no private account material is part of this report.

## 9. Revision and research plan

The user requested a report that can be revised within an available timeframe,
but no deadline or cadence was supplied. The plan below is therefore ordered by
dependency and can be time-boxed without inventing a date.

### Stage A: make the current draft reproducible

1. Update or label docs/architecture.md so removed local-BM25/FetchTool
   behavior is clearly historical.
2. Record exact dependency resolution, application commit, query-kit commit,
   model/provider configuration, and container image versions.
3. [Completed for this snapshot] Start PostgreSQL and Redis with the declared
   Compose configuration, apply migrations, rerun integration tests, and execute
   the deterministic smoke path.
4. [Completed for this snapshot] Run the Chainlit auth/persistence and streaming
   smoke tests on an alternate host port; save the port-conflict limitation.

### Stage B: establish a small evidence benchmark

1. Collect and review a stratified set of real Indic research questions.
2. Define graded relevance and claim-support labels with at least two reviewers
   for a pilot subset.
3. Capture provider results and replay metadata.
4. Compare original queries, normalized keyword queries, provider-friendly
   fallbacks, and tool budgets under the same model/configuration.
5. Report retrieval, tool-process, grounding, latency, cost, and failure
   metrics with uncertainty and per-language breakdowns.

### Stage C: publication-grade evaluation

1. Add a controlled human evaluation of answer correctness, source usefulness,
   citation faithfulness, and language/script fidelity.
2. Run ablations for tool-call budget, provider choice, query normalization,
   cache behavior, and prompt version.
3. Only if the baseline shows a measured gap, create a separate approved plan
   for semantic retrieval or hybrid retrieval. That plan should compare against
   the keyword baseline and include embedding/model/index cost and privacy.
4. Evaluate prompt injection, malicious or misleading sources, provider failure,
   stale results, and data-retention behavior.
5. Release a benchmark manifest, annotation guidelines, result captures, exact
   environment, and limitations statement.

The result of Stage A would support a robust technical report. Stages B and C
are needed before making broad claims about effectiveness or research
productivity.

## 10. Conclusion

indic-research-agent is currently a coherent, bounded research-assistant
scaffold: Chainlit provides a thin streaming interface; a controller and service
layer connect the request to a LangGraph tool loop; LiteLLM supplies the model
boundary; a typed search tool calls public providers through query-kit; and
events, traces, persistence, and caching make the workflow observable and
repeatable enough for further study. The architecture deliberately excludes
embeddings, vector databases, and local document search at this stage.

The evidence supports a narrower conclusion than a product launch claim. The
repository passes formatting, lint, unit, integration, deterministic smoke, and
selected Chainlit e2e gates; the independent query-kit component passes its
suite and returns live Semantic Scholar results for a Hindi OCR query. No
retrieval, grounding, latency, cost, or user benchmark has yet been executed.
Therefore, the project has demonstrated a working boundary and a testable
orchestration design, not yet a measured research advantage.

That distinction gives the project a credible next step. Freeze the current
runtime and documentation, complete service/browser validation, build a small
Indic-language benchmark, and evaluate the full chain from query formation to
source-supported answer. If keyword-first retrieval is insufficient on measured
tasks, the project will have evidence for a targeted extension. Until then, the
simple, inspectable path is the correct scientific baseline.

## References

1. Stephen Robertson and Hugo Zaragoza. “The Probabilistic Relevance
   Framework: BM25 and Beyond.” *Foundations and Trends in Information
   Retrieval*, 2009. [DOI](https://doi.org/10.1561/1500000019).
2. Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. *Introduction
   to Information Retrieval*, Okapi BM25 chapter. [Stanford IR Book](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).
3. Patrick Lewis et al. “Retrieval-Augmented Generation for Knowledge-Intensive
   NLP Tasks.” 2020. [arXiv](https://arxiv.org/abs/2005.11401).
4. Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu.
   “Evaluation of Retrieval-Augmented Generation: A Survey.” 2024.
   [HTML paper](https://arxiv.org/html/2405.07437v2).
5. Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. “Ragas:
   Automated Evaluation of Retrieval Augmented Generation.” 2023.
   [arXiv](https://arxiv.org/abs/2309.15217).
6. Shunyu Yao et al. “ReAct: Synergizing Reasoning and Acting in Language
   Models.” 2022. [arXiv](https://arxiv.org/abs/2210.03629).
7. Derek H. Mohammadi et al. “Evaluation and Benchmarking of LLM Agents: A
   Survey.” 2025. [arXiv](https://arxiv.org/abs/2507.21504) and
   [ACM record](https://doi.org/10.1145/3711896.3736570).
8. “A Survey on LLM Agent Evaluation.” 2025.
   [arXiv](https://arxiv.org/abs/2503.16416).
9. Abhijit Mishra et al. “A Survey of NLP Resources, Tools, and Techniques for
   Indian Languages.” 2022. [ACM record](https://doi.org/10.1145/3548457).
10. Divyanshu Kakwani et al. “IndicNLPSuite: Monolingual Corpora, Evaluation
    Benchmarks and Pre-trained Multilingual Language Models for Indian
    Languages.” Findings of EMNLP, 2020. [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.445/).
11. Akshat Raja and Anshuman Vats. “Parallel Corpora for Machine Translation in
    Low-Resource Indic Languages: A Comprehensive Review.” 2025.
    [ACL Anthology](https://aclanthology.org/2025.loresmt-1.12/).
12. K. Ghosh et al. “SafeSpeech: a three-module pipeline for hate intensity
    mitigation of social media texts in Indic languages.” 2024.
    [Springer article](https://link.springer.com/article/10.1007/s13278-024-01393-9).
13. LangChain. “LangGraph streaming.” [Official documentation](https://docs.langchain.com/oss/python/langgraph/streaming/).
14. LiteLLM. “Streaming.” [Official documentation](https://docs.litellm.ai/docs/completion/stream).
15. Harry Percival and Bob Gregory. *Architecture Patterns with Python*, service
    layer chapter. [Cosmic Python](https://www.cosmicpython.com/book/chapter_04_service_layer.html).
16. Harry Percival and Bob Gregory. *Architecture Patterns with Python*,
    dependency injection chapter. [Cosmic Python](https://www.cosmicpython.com/book/chapter_13_dependency_injection.html).
17. query-kit, public provider-query component. [GitHub repository](https://github.com/neeraj1909/query-kit).
18. AI4Bharat. “IndicBERT.” [Official model page](https://ai4bharat.iitm.ac.in/areas/model/LLM/IndicBERT/).

## Appendix A. Repository evidence map

| Evidence | Role in this report |
|---|---|
| [README.md](../README.md) | Current product scope, architecture constraints, and explicit absence of local document search/vector retrieval. |
| [pyproject.toml](../pyproject.toml) | Python/dependency/runtime declaration. |
| [agent/graph.py](../src/indic_research_agent/agent/graph.py) | Graph nodes, routing, tool budget, tracing, finalization. |
| [agent/prompts.py](../src/indic_research_agent/agent/prompts.py) | Scope, search, citation, and runtime-boundary policy. |
| [tools/schemas.py](../src/indic_research_agent/tools/schemas.py) | Typed search input and result contracts. |
| [tools/search.py](../src/indic_research_agent/tools/search.py) | Search execution, normalization, and fallback behavior. |
| [services/querykit_service.py](../src/indic_research_agent/services/querykit_service.py) | External query-kit/provider boundary. |
| [ui/chainlit_stream_renderer.py](../src/indic_research_agent/ui/chainlit_stream_renderer.py) | Event-to-Chainlit mapping and ephemeral status lifecycle. |
| [services/cache_service.py](../src/indic_research_agent/services/cache_service.py) | Redis cache key/TTL behavior. |
| [tests/unit/](../tests/unit/) | Fast implementation and architecture guard evidence. |
| [scripts/smoke_query.py](../scripts/smoke_query.py) | Deterministic application smoke entry point for service-enabled validation. |
| [compose.yaml](../compose.yaml) | PostgreSQL/Redis/app service topology. |

## Appendix B. Reproduction notes

To reproduce the local gates for this snapshot, run from the repository root:

~~~bash
uv run ruff format --check .
uv run ruff check .
timeout 60 uv run pytest -m unit --no-cov
timeout 60 uv run pytest -m integration --no-cov
uv run pytest -m e2e
~~~

The integration and e2e commands require the configured external services and
environment. For the independent provider boundary:

~~~bash
cd /home/neeraj/Code/query-kit
timeout 60 uv run pytest
timeout 60 uv run query-cli search "Hindi OCR" --provider semantic-scholar --limit 3 --format json
~~~

The exact output, provider state, model configuration, and service logs should
be saved with any future benchmark run. The current report intentionally records
the absence of those artifacts rather than filling the gap with estimates.
