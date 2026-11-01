# FRAMES benchmark comparison slice

This directory contains a small, local, deterministic slice of the published
FRAMES test split for the Decisions-enabled versus Decisions-bypassed
comparison. It is not a relabeled or paraphrased fixture.

## Artifact

- File: frames-test-20.jsonl
- Rows: 20
- Dataset: google/frames-benchmark, test split
- Dataset revision: 58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef
- Official source file: test.tsv
- Official source SHA-256: 4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff
- Local JSONL SHA-256: 4d7c97d9fa7ef263c11c26afab7f75ec4b954641b5c04e9e531f0b3520b3918a
- License reported by the official dataset card: Apache-2.0

Official sources:

- Dataset card/API: https://huggingface.co/datasets/google/frames-benchmark
- Pinned TSV: https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv

## Selection and schema

Selection is exactly source rows 0 through 19 in the pinned official TSV, in
source order. IDs are stable local keys of the form frames-test-rNNNN,
derived from the official zero-based row index. No semantic-equivalence
assumption or answer normalization is applied.

Every row has the minimal fields id, question, reference_answer,
evidence_urls, and source. official_fields preserves the original TSV fields,
including the original prompt/answer strings, numbered Wikipedia links,
reasoning_types, and the source's wiki_links string.

## Integrity and local use

The build check verified 20 JSONL rows, 20 unique IDs, 20 unique source-row
indices, and zero empty evidence_urls arrays. The JSONL is intended to be
read directly from the repository; no fresh download is required for a local
comparison run.

Example:

    uv run python - <<'PY'
    import json
    from pathlib import Path
    rows = [json.loads(line) for line in Path(
        'data/evaluations/frames/frames-test-20.jsonl'
    ).read_text().splitlines()]
    print(len(rows), rows[0]['id'], rows[-1]['id'])
    PY

## Limitations

FRAMES is an English, Wikipedia-grounded, multi-hop benchmark. This slice is
only the first 20 official test rows, not a statistically representative
sample. The current application's configured research providers may not expose
the referenced Wikipedia evidence, so comparison results can conflate routing,
provider coverage, retrieval, and reasoning. Report paired item outcomes and
provider failures separately; do not infer semantic equality from exact answer
strings and do not present this slice as evidence of broad Indic-language
quality.
