# Token-Probability Figures

This directory contains public-safe figures derived from aggregate teacher-forcing token-probability tables for `Qwen/Qwen2.5-7B-Instruct`.

The four aggregate figures use final dialogue responses, `subset=all`, and 95% bootstrap confidence intervals from `tables/qwen_token_probability_summary.csv`. The public paired-test and correlation tables are validated structurally but are not substitutes for scenario-level paired observations.

## Current Status

Four aggregate figures can be reproduced from public data. The paired polite-versus-direct figure and harmfulness-delta scatter remain blocked because the private pair input is not present locally. They are intentionally not generated from aggregate paired-test rows.

## Public-Only Generation

```bash
python -m src.plot_token_probabilities \
  --summary-input tables/qwen_token_probability_summary.csv \
  --paired-tests-input tables/qwen_token_probability_paired_tests.csv \
  --correlations-input tables/qwen_token_probability_correlations.csv \
  --output-dir figures/token_probabilities \
  --manifest-output figures/token_probabilities/figure_manifest.json
```

This command exits with status code 2 after producing the four public aggregate figures and a manifest with status `BLOCKED_PRIVATE_INPUT`.

## Private Colab Generation

Run the following command only in the private Colab environment where the pair file was produced. Keep that CSV outside the repository and do not upload or commit it.

```bash
python -m src.plot_token_probabilities \
  --summary-input tables/qwen_token_probability_summary.csv \
  --paired-tests-input tables/qwen_token_probability_paired_tests.csv \
  --correlations-input tables/qwen_token_probability_correlations.csv \
  --private-pairs-input /content/private/qwen_token_probability_pair_deltas.csv \
  --output-dir figures/token_probabilities \
  --manifest-output figures/token_probabilities/figure_manifest.json
```

The private input may contain scenario-level identifiers and paired values. The plotting code uses only safe numeric fields and labels, never copies the private CSV, and does not place `scenario_id`, prompts, responses, tokens, or model-output text into public figures or the manifest.
