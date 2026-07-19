# Token-Probability Figures

This directory contains six public-safe teacher-forcing token-probability figures for `Qwen/Qwen2.5-7B-Instruct`.

## Aggregate Figures

The four aggregate figures use final dialogue outputs, `subset=all`, and 95% bootstrap confidence intervals from `tables/qwen_token_probability_summary.csv`:

- `qwen_mean_token_logprob.png`
- `qwen_geometric_mean_probability.png`
- `qwen_first16_logprob.png`
- `qwen_entropy.png`

## Paired Figures

The two paired figures were generated locally in Colab from the private scenario-level pair file:

- `qwen_polite_direct_paired.png`
- `qwen_harmfulness_delta_scatter.png`

Only the verified public PNG files and aggregate manifest are stored in this repository. The private CSV remains outside the repository and must not be committed.

## Reproduction In Colab

Mount the private Google Drive location, keep the CSV outside the Git checkout, and run:

```bash
python -m src.plot_token_probabilities \
  --summary-input tables/qwen_token_probability_summary.csv \
  --paired-tests-input tables/qwen_token_probability_paired_tests.csv \
  --correlations-input tables/qwen_token_probability_correlations.csv \
  --private-pairs-input /content/drive/MyDrive/Deviant-AI-private/results/qwen_token_probability_pair_deltas.csv \
  --output-dir figures/token_probabilities \
  --manifest-output figures/token_probabilities/figure_manifest.json
```

The plotting pipeline reads only the numeric fields needed for the paired visualizations and does not copy the private CSV into the repository.
