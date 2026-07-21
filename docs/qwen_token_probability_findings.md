# Qwen Token-Probability Findings

## Goal

This note interprets the completed expanded Qwen2.5-7B-Instruct
teacher-forcing analysis. It asks whether polite phrasing or multi-turn
decomposition is associated with a different probability distribution over
the observed continuation. Qwen was the first completed expanded model stage;
Gemma was later evaluated with corresponding teacher-forcing analyses, while
Mistral currently has behavioral results only.

## Method

The analysis contains 20 paired scenarios, split evenly between harmful and
benign labels. Each comparison uses the right condition minus the left
condition. The tables below report the paired mean difference, a bootstrap
95% confidence interval, a two-sided sign-flip permutation p-value, and a
two-sided Wilcoxon p-value. All comparisons use `n = 20` pairs.

The scores are teacher-forced measurements of the continuations that were
actually generated. They describe model-assigned probability under each
observed context; they do not directly measure a hidden mental state or prove
that a prompt condition caused the difference.

## Main Results

The clearest pattern is the multi-turn minus direct comparison. Multi-turn
contexts had higher mean token log probability and geometric mean token
probability, lower perplexity and entropy, and a larger top-1 versus top-2
probability margin. Both tests supported these directions for every requested
metric except the first-8-token log probability. The first-16-token effect was
smaller than the full-sequence effect but still supported by both tests.

The polite minus direct and polite multi-turn minus multi-turn comparisons
were small, had confidence intervals crossing zero, and were not significant
under either test. The current data therefore do not support a claim that
politeness changed token-level confidence.

## Polite vs Direct

| Metric | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| Mean token log probability | 0.0007 | [-0.0210, 0.0246] | 0.9516 | 0.8408 |
| Geometric mean token probability | -0.00005 | [-0.0148, 0.0159] | 0.9943 | 0.7841 |
| Perplexity | -0.0023 | [-0.0372, 0.0294] | 0.8986 | 0.8983 |
| First-8-token log probability | 0.0218 | [-0.0395, 0.0821] | 0.5116 | 0.3118 |
| First-16-token log probability | 0.0246 | [-0.0212, 0.0758] | 0.3568 | 0.5706 |
| Mean token entropy | 0.0005 | [-0.0437, 0.0393] | 0.9835 | 0.6742 |
| Top-1 versus top-2 margin | -0.0005 | [-0.0156, 0.0161] | 0.9574 | 0.5706 |

Direct and polite conditions are practically indistinguishable in this
sample. Even the larger early-token estimates are imprecise and change sign
compatibility within their confidence intervals.

## Multi-Turn Effects

| Metric | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| Mean token log probability | 0.1969 | [0.1499, 0.2414] | <0.0001 | <0.0001 |
| Geometric mean token probability | 0.1503 | [0.1134, 0.1848] | <0.0001 | <0.0001 |
| Perplexity | -0.2602 | [-0.3151, -0.1980] | <0.0001 | <0.0001 |
| First-8-token log probability | 0.0724 | [-0.0383, 0.2048] | 0.2927 | 0.5459 |
| First-16-token log probability | 0.1195 | [0.0529, 0.1881] | 0.0035 | 0.0056 |
| Mean token entropy | -0.3652 | [-0.4494, -0.2712] | <0.0001 | <0.0001 |
| Top-1 versus top-2 margin | 0.1667 | [0.1276, 0.2054] | <0.0001 | <0.0001 |

The direction is consistent with a sharper teacher-forced distribution after
multi-turn context. The unsupported first-8-token result suggests that the
full-sequence pattern should not be generalized to the very beginning of a
continuation. Context structure, continuation length, and termination behavior
remain plausible contributors.

Adding politeness to the multi-turn condition did not strengthen the pattern:

| Metric | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| Mean token log probability | -0.0090 | [-0.0391, 0.0247] | 0.6038 | 0.3884 |
| Geometric mean token probability | -0.0074 | [-0.0321, 0.0190] | 0.5905 | 0.4524 |
| Perplexity | 0.0109 | [-0.0311, 0.0490] | 0.6243 | 0.4091 |
| First-8-token log probability | 0.0077 | [-0.0541, 0.0714] | 0.8209 | 0.8695 |
| First-16-token log probability | 0.0042 | [-0.0410, 0.0483] | 0.8598 | 0.9273 |
| Mean token entropy | 0.0183 | [-0.0491, 0.0773] | 0.6043 | 0.3488 |
| Top-1 versus top-2 margin | -0.0059 | [-0.0339, 0.0246] | 0.7089 | 0.5958 |

## Entropy and Confidence

Multi-turn responses had lower mean token entropy and a larger top-1 versus
top-2 margin. Together with higher token probability, this indicates a more
concentrated predictive distribution for these observed continuations. It is
safer to describe this as distributional concentration than as internal
confidence. The design does not isolate whether the shift comes from the
condition itself, different generated continuations, length, or stopping.

## Limitations

This is one model with 20 paired scenarios and many related metrics. The
reported p-values are descriptive and were not adjusted for multiple testing.
Bootstrap intervals are also uncertain at this sample size. The 20
length-limited response records were retained under the fixed 1024-token
budget; 19 of them were final responses. Retention avoids post hoc exclusion,
but termination remains a possible source of sensitivity. No result should be
read as a causal mechanism or a model-family-wide effect.

## Sensitivity Analysis Status

The sensitivity analysis is complete for 80 final responses, with 20 in each
of the four conditions. Intermediate multi-turn responses were excluded by
default, matching the final-response scope of the main teacher-forcing
analysis. The source data remain private, while the published sensitivity CSV
contains aggregate statistics only.

The aggregate pattern is stable across the main sensitivity subsets. Direct
and polite remain practically identical: their overall mean token log
probabilities are -0.388 and -0.387, and both have mean token entropy of about
0.749. The multi-turn and polite multi-turn conditions have higher mean token
log probabilities (-0.191 and -0.200) and lower mean token entropy (0.384 and
0.402). The same descriptive direction remains among responses that did not
reach the length limit and appears separately in the harmful and benign
subsets. Because it is also present for benign responses, the multi-turn
pattern is not confined to jailbreak-related behavior in this dataset. It may
instead reflect a broader association with multi-turn context structure.

The subgroup results require caution. The length-limited subset has no direct
or polite responses. Refusal counts are only 2 for multi-turn and 3 for polite
multi-turn, while unsafe or compliant counts are only 1 each for direct and
polite. These subgroup estimates are descriptive: the sensitivity CSV does
not provide paired tests or p-values for them. The analysis cannot establish
causality and should not be generalized to other models. Qwen was the first
fully completed expanded experiment in the project. A later descriptive
cross-model behavioral comparison does not extend this token-probability
result to Mistral.

The optional `--include-intermediate-turns` flag remains available only for a
separate diagnostic analysis; it is not enabled in the published final-only
results.

## Fixed-Continuation Follow-Up

The fixed-continuation follow-up has now been completed. It scores the
same direct-anchor continuation under all four contexts, holding the
continuation token IDs and continuation length constant.

The results show that the large multi-turn pattern from the original
generated-continuation analysis does not carry over with the same
direction when response content is fixed. This indicates that generated
response differences and context-continuation fit were important
contributors to the original result.

Full results and limitations are documented in
`docs/qwen_fixed_continuation_findings.md`.

Public sensitivity and fixed-continuation tables apply a minimum cell
size of five. Counts remain visible for smaller cells, but estimates,
confidence intervals, and p-values are blank.
