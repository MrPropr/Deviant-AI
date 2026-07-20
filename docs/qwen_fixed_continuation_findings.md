# Qwen Fixed-Continuation Findings

## Goal

This follow-up isolates context effects by scoring the exact same
direct-condition continuation under four prompt contexts:

- direct;
- polite;
- multi-turn;
- polite multi-turn.

The continuation token IDs and continuation length are identical across
the four conditions within every scenario.

## Data and Method

The analysis contains 20 paired scenarios and 80 context-continuation
scores from Qwen2.5-7B-Instruct. Each comparison is calculated as the
right condition minus the left condition.

Detailed prompts, continuations, scenario identifiers, token IDs, and
raw model outputs remain private. Public cells smaller than five
observations retain their count but have statistical estimates
suppressed.

## Main Results

| Comparison | Metric | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: |
| Polite minus direct | Mean token log probability | -0.0100 | [-0.0130, -0.0071] | <0.0001 | <0.0001 |
| Polite minus direct | Mean token entropy | 0.0178 | [0.0153, 0.0203] | <0.0001 | <0.0001 |
| Multi-turn minus direct | Mean token log probability | -0.3462 | [-0.4222, -0.2768] | <0.0001 | <0.0001 |
| Multi-turn minus direct | Mean token entropy | -0.1198 | [-0.1398, -0.1003] | <0.0001 | <0.0001 |
| Polite multi-turn minus multi-turn | Mean token log probability | 0.0041 | [-0.0210, 0.0307] | 0.7664 | 0.5706 |
| Polite multi-turn minus multi-turn | Mean token entropy | -0.0145 | [-0.0331, 0.0015] | 0.1386 | 0.4980 |

## Interpretation

Under the single-turn condition, polite phrasing produced a small but
consistent reduction in the probability assigned to the fixed direct
continuation and a small increase in predictive entropy.

The multi-turn context produced the largest effect. It assigned much
lower probability to the fixed direct continuation while also producing
a more concentrated next-token distribution. This combination suggests
that the multi-turn context pushes the model confidently toward a
different continuation rather than making the direct continuation
generally easier to predict.

Adding politeness to the multi-turn context did not produce a reliable
change in either primary metric.

The direction of the fixed-continuation multi-turn result differs from
the earlier generated-continuation analysis. This indicates that the
earlier increase in token probability was substantially associated with
differences in the generated continuations and their fit to their own
contexts, rather than a context-only increase for identical text.

## Limitations

The continuation is anchored to the direct-condition response. It may
therefore represent refusals more often in harmful scenarios and does not
symmetrically sample continuations from all four conditions.

The experiment contains one model and 20 paired scenarios. P-values are
descriptive and are not adjusted for multiple related metrics. Results
should not be interpreted as a hidden-state measurement or generalized
to other model families without replication.
