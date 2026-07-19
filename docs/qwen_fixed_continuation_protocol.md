# Qwen Fixed-Continuation Protocol

## Purpose

This experiment tests whether Qwen2.5-7B-Instruct assigns different
teacher-forced probabilities to the same continuation when only the preceding
conversation structure changes. The four contexts are `direct`, `polite`,
`multi_turn`, and `polite_multi_turn`. Qwen is the first model used for this
stage of the project, not the project's only model.

The experiment isolates context sensitivity from differences among responses
that a model would generate independently. It performs no response generation,
sampling, adaptive optimization, or tool use.

## Direct-Anchor Design

Each scenario has one fixed continuation taken privately from the final
response in the `direct` condition. That exact continuation is scored under all
four contexts. Holding the continuation fixed means that any measured
difference is associated with the preceding context rather than a change in
the scored response text.

Before scoring, the pipeline applies the tokenizer chat template with
`add_generation_prompt=True`. It tokenizes the context prefix and the complete
prefix-plus-continuation input separately, verifies that the prefix token IDs
are an exact prefix of the complete input, and extracts the continuation as the
remaining suffix. Scoring proceeds only when the continuation suffix contains
at least one token and its token IDs are exactly identical in all four
conditions. The pipeline does not silently align, truncate, or replace tokens.

Only continuation tokens contribute to the reported metrics. Context tokens
are used to establish the model state but are not included in the metric
calculations.

## Scoring Controls

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: bitsandbytes 4-bit NF4
- Compute dtype: float16
- Inference: deterministic teacher forcing
- Seed: 42
- Generation and sampling: disabled
- Resume key: `scenario_id` plus condition

The private detail output is atomically checkpointed after every completed
condition. A resumed run validates existing metadata and token counts and does
not duplicate completed records.

## Data Handling

The input JSONL and detail score JSONL are private artifacts. They must remain
outside Git or under an ignored private repository directory. Console output
reports counts and status only; it does not print contexts, continuations,
token IDs, or model text.

The only publishable outputs are aggregate CSV tables. The summary table
contains subset, condition, metric, sample size, mean, standard deviation, and
bootstrap confidence bounds. The paired table contains aggregate comparisons,
sample sizes, mean right-minus-left differences, bootstrap confidence bounds,
and two-sided sign-flip and Wilcoxon p-values. Neither public table contains
scenario-level rows or paired differences.

## Colab Commands

Use a private Colab directory that is not part of the repository. First validate
the schema and token boundary without loading the model:

```bash
python -m src.score_fixed_continuations \
  --input /content/private/qwen_fixed_continuation_input.jsonl \
  --validate-only
```

For an offline plumbing check with the synthetic backend:

```bash
python -m src.score_fixed_continuations \
  --input /content/private/qwen_fixed_continuation_input.jsonl \
  --output /content/private/qwen_fixed_continuation_dry_run.jsonl \
  --dry-run \
  --seed 42
```

Run the fixed-continuation scoring job:

```bash
python -m src.score_fixed_continuations \
  --input /content/private/qwen_fixed_continuation_input.jsonl \
  --output /content/private/qwen_fixed_continuation_scores.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --seed 42
```

Add `--resume` to that command after an interruption. Once private scoring is
complete, create only the public aggregate tables:

```bash
python -m src.analyze_fixed_continuations \
  --input /content/private/qwen_fixed_continuation_scores.jsonl \
  --summary-output tables/qwen_fixed_continuation_summary.csv \
  --paired-output tables/qwen_fixed_continuation_paired_tests.csv \
  --seed 42
```

## Limitations

The direct-anchor continuation may reflect features specific to responses from
the direct condition. For harmful scenarios, this may produce a collection with
many refusals. The experiment therefore measures the context sensitivity of a
fixed direct continuation; it does not represent every possible harmful or
compliant response type.

Teacher-forced probability differences are descriptive model-behavior signals,
not proof of a causal mechanism. Results from Qwen2.5-7B-Instruct should not be
generalized to other models without replication. Quantization, tokenizer
behavior, sample composition, and small subgroup sizes may also affect the
observed estimates.
