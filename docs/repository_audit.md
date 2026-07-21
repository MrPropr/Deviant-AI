# Repository Audit

## Result

**PASS**, with the reproducibility and study-design limitations documented below.

This public-safe audit covers the repository from its first commit through the
verified base commit, plus the narrowly scoped corrections made during the
audit. It reviews repository structure, history, code, tests, configurations,
documentation, aggregate tables, figures, schemas, the public notebook, links,
privacy controls, and credential hygiene. It does not claim to review external
chat sessions, notebook runtime output that was not committed, or other
off-repository activity.

## Scope and Inventory

- Verified base commit: `c3db60ebc9e49f57dbc76df3bf045c71622d3324`.
- Base tracked files reviewed: 115.
- Final public files after adding this report: 116.
- Historical commits reviewed before the audit commit: 30.
- Markdown files reviewed after adding this report: 17.
- CSV files reviewed: 24.
- PNG figures reviewed: 27.
- Configuration files reviewed: 6.
- Python files reviewed: 31.
- Python test files reviewed: 12.
- Notebook files reviewed: 1.
- Tracked GitHub workflow files: 0.

All tracked directories and files were inventoried. No empty files, broken
symbolic links, accidental archives, notebook checkpoints, backup files, logs,
or temporary artifacts were found. Identical empty directory markers are the
only duplicate file contents. Tracked text uses LF in the Git index and has a
terminal newline.

## Repository History

The audit reviewed each of the 30 historical commits, its message, and its file
change summary. The changes align with the repository's documented progression
from scaffold, through the Qwen stage, teacher-forcing analysis, Gemma and
Mistral stages, and cross-model reporting. Two merge commits contain no direct
file changes. No tags or tracked Actions workflows exist. Two pull requests,
zero issues, and zero workflow runs were accessible through the public
repository interface at audit time.

No credential-like value or tracked raw/private path was found in the reviewed
history. The retired-model text scan found 87 historical line matches across
two public paths and no historical filename matches. History was not rewritten.
The current-tree retired-model scan has zero text matches and zero filename or
directory-name matches.

## Issues Found and Fixed

1. The README and model registry mixed completed work with stale model,
   benchmark, API, and analysis plans.
2. Several reports described Qwen as the only completed expanded stage or
   described cross-model work as still pending.
3. The experiment log omitted completed public Qwen, Gemma, Mistral, and
   cross-model milestones.
4. Package metadata claimed Python 3.9 support even though the codebase uses
   Python 3.10 syntax.
5. Aggregate analysis accepted incomplete or duplicate multi-turn sequences.
6. Validation errors could echo record-level identifiers from private inputs.
7. Output path validation incorrectly rejected safe off-repository private
   locations in some scripts.
8. Three output-producing scripts lacked the same public-path protection used
   by the rest of the pipeline.
9. One analysis module contained an unused type import.

The corrections remove stale planning language and current retired-model
references, document the completed stages, require Python 3.10 or newer,
exclude malformed conversation sequences, use row-number-only diagnostics,
allow off-repository outputs, enforce ignored paths for sensitive outputs kept
inside the checkout, and remove the dead import. Five focused regression tests
cover these fixes. No published CSV, image, manifest, notebook, schema, or
experimental result value was changed.

## README Review

The README now describes the completed fixed dataset scope and the three
completed open-weight model stages. Qwen remains explicitly identified as the
first completed expanded experiment, not the project's only model. The
cross-model comparison remains descriptive, and no closed-source result is
claimed. The former planned-figures section now describes published artifacts.
The project checklist contains exactly one unfinished item: the final paper
manuscript.

All 51 local Markdown links resolve. Every reviewed Markdown document has one
top-level heading, no heading-level jumps, no duplicate headings, and no broken
links to public tables, figures, configurations, or reports.

## Pipeline Review

The end-to-end public code path was reviewed for dataset preparation, fixed
four-condition template handling, validation, deterministic local generation,
placeholder API behavior, annotation-sheet creation, scoring, aggregation,
statistical summaries, plotting, resume behavior, and context isolation.

The completed model configurations use the documented model identifiers,
deterministic decoding, recorded seeds, and three-turn maximum for both
multi-turn conditions. Large-model loading remains explicit and opt-in. Raw
text is not printed by the model runner. Incremental writes, resume checks,
generation termination metadata, interruption handling, and CUDA memory error
handling remain covered by tests.

The public notebook is a sanitized three-cell scaffold with no stored outputs.
The JSON schema and sanitized example file use placeholders only. Repository
agent instructions and ignore rules consistently require aggregate-only public
artifacts.

## Numerical Audit

All 24 CSV files decode as UTF-8, have nonempty unique headers, contain no
duplicate rows, and contain finite numeric values where required. Rates and
scores remain within their declared ranges. Model labels, condition labels,
sample counts, and effect definitions are internally consistent. No record-level
text or identifiers are present in public table headers.

The 12 cross-model metric rows agree with the three model-specific metric
tables. The 12 cross-model effect rows recompute from their corresponding
condition values. The 52 published Qwen and Mistral graph-source rows agree with
their model metric tables. The primary final-output token-probability summaries
contain 20 observations for each of the four conditions. No numerical
discrepancy was found.

## Figure Audit

Every image opens and verifies as PNG through Pillow. Embedded metadata contains
no private text or identifiers. Titles, axes, legends, and condition labels are
legible; no clipping, overlap, misleading footer, or blank plot was found during
visual review. Hashes for the six Qwen teacher-forcing figures agree with their
manifest. The table records the source class and verification level for every
published figure.

| Figure | Pixels | DPI | Source | Values | Visual/privacy |
| --- | ---: | ---: | --- | --- | --- |
| `figures/cross_model_benign_success.png` | 1848x1094 | 180 | Cross-model metric table | Verified | PASS |
| `figures/cross_model_harmfulness.png` | 1871x1094 | 180 | Cross-model metric table | Verified | PASS |
| `figures/cross_model_multi_turn_effects.png` | 1871x1094 | 180 | Cross-model effect table | Verified | PASS |
| `figures/cross_model_partial_compliance.png` | 1871x1094 | 180 | Cross-model metric table | Verified | PASS |
| `figures/cross_model_refusal_rate.png` | 1871x1094 | 180 | Cross-model metric table | Verified | PASS |
| `figures/cross_model_strict_asr.png` | 1871x1094 | 180 | Cross-model metric table | Verified | PASS |
| `figures/gemma_expanded_asr.png` | 1620x990 | 180 | Gemma behavioral metric table | Verified | PASS |
| `figures/gemma_expanded_benign_success.png` | 1620x990 | 180 | Gemma behavioral metric table | Verified | PASS |
| `figures/gemma_expanded_harmfulness.png` | 1620x990 | 180 | Gemma behavioral metric table | Verified | PASS |
| `figures/gemma_expanded_refusal.png` | 1620x990 | 180 | Gemma behavioral metric table | Verified | PASS |
| `figures/gemma_fixed_continuation_harmful_paired.png` | 2700x990 | 180 | Public Gemma paired table | Verified | PASS |
| `figures/gemma_token_probability_harmful_paired.png` | 2700x990 | 180 | Public Gemma paired table | Verified | PASS |
| `figures/mistral_expanded_asr.png` | 1620x990 | 180 | Mistral graph-value table | Verified | PASS |
| `figures/mistral_expanded_benign_success.png` | 1620x990 | 180 | Mistral graph-value table | Verified | PASS |
| `figures/mistral_expanded_discovery_curve.png` | 1511x972 | 180 | Mistral graph-value table | Verified | PASS |
| `figures/mistral_expanded_harmfulness.png` | 1620x990 | 180 | Mistral graph-value table | Verified | PASS |
| `figures/mistral_expanded_refusal.png` | 1620x990 | 180 | Mistral graph-value table | Verified | PASS |
| `figures/qwen_expanded_asr.png` | 1620x990 | 180 | Qwen graph-value table | Verified | PASS |
| `figures/qwen_expanded_discovery_curve.png` | 1440x990 | 180 | Qwen graph-value table | Verified | PASS |
| `figures/qwen_expanded_harmfulness.png` | 1620x990 | 180 | Qwen graph-value table | Verified | PASS |
| `figures/qwen_expanded_refusal.png` | 1620x990 | 180 | Qwen graph-value table | Verified | PASS |
| `figures/token_probabilities/qwen_entropy.png` | 2160x1440 | 300 | Public aggregate summary | Manifest and values verified | PASS |
| `figures/token_probabilities/qwen_first16_logprob.png` | 2160x1440 | 300 | Public aggregate summary | Manifest and values verified | PASS |
| `figures/token_probabilities/qwen_geometric_mean_probability.png` | 2160x1440 | 300 | Public aggregate summary | Manifest and values verified | PASS |
| `figures/token_probabilities/qwen_harmfulness_delta_scatter.png` | 2160x1440 | 300 | Reviewed off-repository pair input | Manifest/hash verified | PASS |
| `figures/token_probabilities/qwen_mean_token_logprob.png` | 2160x1440 | 300 | Public aggregate summary | Manifest and values verified | PASS |
| `figures/token_probabilities/qwen_polite_direct_paired.png` | 2160x1440 | 300 | Reviewed off-repository pair input | Manifest/hash verified | PASS |

## Privacy and Credential Audit

No raw harmful content, actionable harmful instructions, raw model text,
record-level identifiers, annotation notes, token-level records, credentials,
or model weights are tracked. All 24 public CSV headers pass the restricted-field
scan. Required sensitive-data locations are ignored and untracked; their
contents were not opened or enumerated. The aggregate count of entries found in
those local locations was zero at audit time. No tracked file is unexpectedly
ignored, and no unexpected untracked file is present apart from this report
before staging.

Credential-pattern scans of both the current tree and Git history found zero
issues. The public data-handling policy, ignore rules, and output-path guards are
consistent with keeping record-level material outside Git.

## Verification

- `python -m compileall -q src tests`: passed.
- `python -m pytest -q`: 106 passed.
- All 17 module CLIs: `--help` passed.
- `git diff --check`: passed.
- Public plotting entry points regenerated 14 source-based PNG files outside
  the checkout: four expanded Qwen, six Gemma, and four aggregate Qwen
  teacher-forcing figures. The aggregate-only teacher-forcing command returned
  its documented missing-private-input status after producing its four public
  figures.
- Markdown link and heading scan: passed.
- CSV structural and numerical scans: passed.
- Pillow decode, metadata, and visual scans: passed.
- Current-tree retired-model scan: 0 text and 0 path matches.
- Historical retired-model scan: 87 line matches, historical only.
- Privacy and credential scans: passed.
- Tracked-ignored and temporary-file scans: passed.

The suite increased from 101 to 106 tests because five regression tests were
added for actual audit fixes: malformed conversation exclusion, identifier-safe
validation diagnostics, safe external output paths, completed model registry,
and identifier-safe resume diagnostics. No test was removed.

## Changed Files

| Path | Reason |
| --- | --- |
| `README.md` | Replace stale plans with verified completed scope and retain one paper item. |
| `configs/models.yaml` | Record only the three completed model stages and remove stale model plans. |
| `docs/experiment_log.md` | Add missing public milestones for completed analyses. |
| `docs/qwen_expanded_pilot.md` | Correct stale project-stage wording without changing findings. |
| `docs/qwen_token_probability_findings.md` | Correct cross-model status and scope wording. |
| `docs/repository_audit.md` | Add this public-safe audit record. |
| `pyproject.toml` | Align declared Python support with syntax used by the codebase. |
| `src/analyze_results.py` | Exclude incomplete, duplicate, or invalid turn sequences. |
| `src/analyze_token_probabilities.py` | Remove a dead import and sanitize validation diagnostics. |
| `src/create_annotation_sheet.py` | Apply consistent safe output-path handling. |
| `src/generate_prompt_variants.py` | Sanitize diagnostics and support safe off-repository output. |
| `src/judge_outputs.py` | Prevent judged records from being written to public repository paths. |
| `src/prepare_dataset.py` | Prevent private dataset rows from being written to public repository paths. |
| `src/run_api_model.py` | Prevent raw API output from being written to public repository paths. |
| `src/run_local_model.py` | Support safe off-repository raw output consistently. |
| `src/score_response_logprobs.py` | Remove record-level identifiers from structural and resume errors. |
| `src/validate_prompt_variants.py` | Centralize safe output-path checks and sanitize diagnostics. |
| `tests/test_analyze_token_probabilities.py` | Cover identifier-safe analysis errors. |
| `tests/test_metrics.py` | Cover exact turn completeness and duplicate-turn exclusion. |
| `tests/test_schemas.py` | Cover safe paths, sanitized errors, and the completed model registry. |
| `tests/test_score_response_logprobs.py` | Cover identifier-safe resume errors. |

No file was deleted. Published result artifacts were not modified.

## Remaining Limitations

- Each expanded model stage uses one fixed seed and 10 harmful plus 10 benign
  scenarios per condition; estimates remain sensitive to individual cases.
- The Qwen stage retained 20 length-limited outputs under its fixed generation
  budget. They were not treated as failures or excluded.
- Annotation procedures differ by stage. The Mistral stage used AI-assisted
  annotation with independent technical validation, not independent human
  reannotation.
- Cross-model comparisons are descriptive and do not establish causality,
  statistical significance, or generalization beyond the tested models.
- No closed-source model result is present, so the full research question is not
  resolved.
- Teacher-forcing analyses are available for Qwen and Gemma, not Mistral.
- Public plotting commands exist for Qwen expanded, Gemma behavioral, and Qwen
  token-probability figures. Dedicated public plotting entry points are absent
  for Mistral and the cross-model figures.
- The plotting dependencies and fonts are not version-pinned. Regenerated
  source-based figures passed numerical and visual checks but were not
  byte-for-byte identical under the audit runtime.
- The two paired Qwen teacher-forcing figures require an off-repository
  record-level source for full regeneration. Their public manifests and hashes
  were verified, but the source is intentionally not stored in Git.
- External chats and uncommitted notebook runtime outputs were not available for
  audit.

These limitations do not block the repository's public safety or internal
consistency, but they constrain scientific interpretation and full public
reproduction of every derived image.
