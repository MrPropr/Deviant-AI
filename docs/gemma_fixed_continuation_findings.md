# Gemma Fixed-Continuation Findings

## Goal

This analysis isolates context sensitivity by scoring the same
direct-condition continuation under direct, polite, multi-turn, and polite
multi-turn contexts for `google/gemma-2-9b-it`.

## Data and Method

The experiment contains 20 scenarios and 80 context-continuation scores. One
direct-anchor continuation is used per scenario. Its continuation token IDs and
length are held constant across all four contexts within that scenario.

Each paired difference is the right condition minus the left condition.
Harmful paired comparisons contain 10 scenarios. Confidence intervals use
10,000 bootstrap iterations, sign-flip tests use 50,000 iterations when exact
enumeration is not used, and the statistical seed is 42.

## Paired Results

| Comparison | Metric | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: |
| Polite minus direct | Mean token log-probability | -0.0005 | [-0.0040, 0.0019] | 0.9062 | 0.5566 |
| Polite minus direct | Perplexity | 0.0006 | [-0.0022, 0.0044] | 0.9062 | 0.5566 |
| Polite minus direct | Mean token entropy | -0.0029 | [-0.0046, -0.0013] | 0.0078 | 0.0098 |
| Multi-turn minus direct | Mean token log-probability | -0.3471 | [-0.4237, -0.2726] | 0.0020 | 0.0020 |
| Multi-turn minus direct | Perplexity | 0.4770 | [0.3588, 0.5992] | 0.0020 | 0.0020 |
| Multi-turn minus direct | Mean token entropy | 0.0604 | [0.0344, 0.0870] | 0.0039 | 0.0039 |
| Polite multi-turn minus multi-turn | Mean token log-probability | 0.0091 | [-0.0215, 0.0406] | 0.5742 | 0.7695 |
| Polite multi-turn minus multi-turn | Perplexity | -0.0193 | [-0.0712, 0.0295] | 0.5254 | 0.7695 |
| Polite multi-turn minus multi-turn | Mean token entropy | -0.0066 | [-0.0157, 0.0018] | 0.1992 | 0.1602 |

The aggregate paired figure is available at
[gemma_fixed_continuation_harmful_paired.png](../figures/gemma_fixed_continuation_harmful_paired.png).

## Interpretation

The multi-turn context assigned substantially lower probability to the same
direct-anchor continuation. Perplexity and token entropy increased at the same
time, showing that the context changed the model's predictive distribution even
though the scored continuation was fixed.

Politeness produced little change in continuation probability or perplexity.
The small polite-minus-direct entropy difference is the only paired politeness
result whose confidence interval excludes zero. It should be interpreted
cautiously because many related metrics were tested and no multiplicity
correction was applied. Adding politeness to the multi-turn condition produced
no reliable effect in the three primary metrics.

Fixed-continuation probability is not a direct harmfulness measure. A lower
probability for the direct anchor does not automatically imply either improved
or reduced safety.

## Limitations

The anchor is taken from the direct response and may be either a refusal or an
unsafe response. The design therefore measures context sensitivity for that
specific direct continuation rather than all possible response types. The
experiment covers one model, one seed, and 20 paired scenarios; the harmful
subset contains 10 pairs. P-values are exploratory and unadjusted. The results
do not establish causality or generalize to other models without replication.

## Privacy

Only aggregate summary and paired-test tables are public. Continuation text,
prompts, model responses, annotations, scenario identifiers, token IDs, and
record-level scores remain outside the repository.
