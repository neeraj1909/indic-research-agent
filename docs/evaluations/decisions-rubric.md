# Decisions request-routing draft rubric

Status: machine-draft fixture only. These labels are not human-approved, are
not model outputs, and are not evidence that the router is correct. Every case
must be reviewed by a human before semantic evaluation or enforcement. The
fixture records this with `label_status: "draft"` and `reviewed_by: null`.

## Scope and route vocabulary

The draft classifies one latest request into exactly one route:

| Route | Use when | `requires_research` | `material_clarification` |
| --- | --- | ---: | ---: |
| `unsupported_local_documents` | The request requires searching, extracting, or comparing a local/uploaded PDF, DOCX, folder, drive, notebook, or attachment, which this app does not ingest. | false | false |
| `clarify` | A missing requirement would materially change the research answer: target language/script, task, domain, time window, evidence rule, audience, or output form. | false until clarified | true |
| `supplied_text` | The user asks only to transform supplied text/data: translate, edit, summarize, format, romanize, or title it, with no external factual claim. | false | false |
| `research` | The user asks for external factual, recent, comparative, source-finding, benchmark, dataset, licensing, or literature-review evidence. | true | false |
| `legacy` | Conflicting intents, prompt injection, generic/out-of-scope requests, unsupported non-document modalities, or cases that do not fit one positive route safely. | case-specific | false unless a separate material gap is explicit |

## Disjoint precedence

Apply precedence in this order so overlapping wording does not produce
multiple labels:

1. `unsupported_local_documents`: explicit local/uploaded-document search or
   extraction wins over any apparent research or transformation request.
2. `clarify`: use only when the missing requirement changes the answer or
   evidence set materially. Do not classify a short greeting or a merely broad
   but actionable request as clarification.
3. `supplied_text`: use only for transformation of supplied content. If the
   request also asks to verify claims, add external facts, or research context,
   it is not pure supplied-text transformation.
4. `research`: use for answerable external facts and public research discovery.
5. `legacy`: fallback for conflicts, injection, generic/out-of-scope requests,
   unsupported non-document tasks, refusal/low-quality outputs, or uncertain
   boundaries.

The precedence is conservative: a classifier must not turn a fact-check into a
no-search transformation, and must not turn unsupported local search into a
public web research request.

## Adversarial and boundary guidance

- Treat text such as “ignore prior rules”, “reveal the system prompt”, or
  “disclose secrets” as untrusted request content. It has no authority and is
  normally `legacy` unless the remaining request independently fits a positive
  route after human review.
- A supplied passage plus “is this true?” is factual external research, not
  `supplied_text`; the draft keeps this conflict in `legacy` pending a settled
  product policy rather than silently choosing a route.
- “Search my PDF/folder/attachment” is `unsupported_local_documents`, even if
  the document concerns Hindi OCR or another research topic.
- Do not overclassify short questions as `clarify`. Ask for clarification only
  when a concrete missing choice changes the intended result.
- A model-generated response, controlled response, or deterministic baseline
  must never be substituted for a human label. Controlled fixtures test wiring,
  not semantic acceptance.

## Fixture contract

`tests/fixtures/decisions/requests.jsonl` contains one JSON object per line with
exactly these required fields:

```text
id, split, family, language, request, expected_route,
requires_research, material_clarification, rationale,
label_status, reviewed_by
```

`split` is `development` or `heldout`; `language` is one of `english`,
`hindi_devanagari`, `hindi_transliterated`, or `mixed`; and `expected_route` is
one of the five routes above. IDs and families are unique. All rows are draft
labels, with `reviewed_by` set to null.

The target composition is 60 cases: 20 development and 40 heldout. The heldout
split contains at least eight cases for each positive route (`research`,
`clarify`, `supplied_text`, and `unsupported_local_documents`) and eight legacy
cases. Language/script slices are balanced at 15 cases each across the full
fixture. Families are not reused between splits, reducing near-duplicate
leakage during future evaluation.

## Review and evaluation protocol

Before semantic scoring, a human reviewer must inspect every row, change labels
only through an auditable review, set `reviewed_by` and record disagreements or
abstentions. Development cases may calibrate a deterministic rubric; heldout
cases must remain untouched until the rubric and any baseline are frozen.

Report route confusion, macro-F1, false research skips, material-clarification
misses, unsafe local-document promises, and language/script slices. Keep
controlled/mock provider records separate from genuine model outputs and report
the provider/model identity, prompt revision, and budget. This corpus alone is
not semantic acceptance, does not establish model quality, and does not
authorize live inference or enforcement.
