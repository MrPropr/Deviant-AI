# Gemma Token-Probability Findings

## Goal

This analysis describes how the probability distribution of
`google/gemma-2-9b-it` varied across direct, polite, multi-turn, and polite
multi-turn conditions. It uses teacher forcing on privately stored generated
responses and publishes aggregate statistics only.

## Data and Method

Teacher forcing was completed for 160 response records. The primary aggregate
analysis uses the 80 final responses, with 20 per condition. Harmful paired
comparisons contain 10 scenario pairs and define each difference as the right
condition minus the left condition.

Confidence intervals use 10,000 bootstrap iterations. Two-sided sign-flip
tests use 50,000 iterations when exact enumeration is not used. The statistical
seed is 42.

## Generated-Continuation Analysis

| Condition | n | Mean token log-probability | Perplexity | Mean token entropy |
| --- | ---: | ---: | ---: | ---: |
| Direct | 20 | -0.1625 | 1.1806 | 0.3456 |
| Polite | 20 | -0.1598 | 1.1768 | 0.3423 |
| Multi-turn | 20 | -0.1816 | 1.2025 | 0.3962 |
| Polite multi-turn | 20 | -0.1882 | 1.2119 | 0.4102 |

These values score each observed response under the context that produced it.
They therefore reflect both context structure and differences among the
generated continuations.

## Paired Comparisons

The following results use harmful scenarios only (`n=10` pairs per cell).

| Comparison | Metric | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: |
| Polite minus direct | Mean token log-probability | -0.0011 | [-0.0127, 0.0077] | 0.8945 | 0.5566 |
| Polite minus direct | Perplexity | 0.0014 | [-0.0084, 0.0147] | 0.8906 | 0.5566 |
| Polite minus direct | Mean token entropy | 0.0032 | [-0.0150, 0.0285] | 0.8945 | 0.5566 |
| Multi-turn minus direct | Mean token log-probability | -0.0570 | [-0.0895, -0.0244] | 0.0137 | 0.0273 |
| Multi-turn minus direct | Perplexity | 0.0672 | [0.0270, 0.1077] | 0.0117 | 0.0195 |
| Multi-turn minus direct | Mean token entropy | 0.1253 | [0.0511, 0.2035] | 0.0117 | 0.0195 |
| Polite multi-turn minus multi-turn | Mean token log-probability | 0.0016 | [-0.0229, 0.0246] | 0.8945 | 0.8457 |
| Polite multi-turn minus multi-turn | Perplexity | 0.0001 | [-0.0293, 0.0310] | 0.9961 | 0.8457 |
| Polite multi-turn minus multi-turn | Mean token entropy | -0.0046 | [-0.0472, 0.0383] | 0.8477 | 0.8457 |

The aggregate paired figure is available at
[gemma_token_probability_harmful_paired.png](../figures/gemma_token_probability_harmful_paired.png).

## Correlations

Across 40 harmful final responses, harmfulness was negatively correlated with
mean token log-probability (Spearman rho = -0.5176, p = 0.0006; bootstrap 95%
CI [-0.6820, -0.2708]) and positively correlated with mean token entropy
(rho = 0.5176, p = 0.0006; CI [0.2708, 0.6822]). The first-16-token
log-probability correlation was weaker and not statistically reliable
(rho = -0.2671, p = 0.0957; CI [-0.5366, 0.1103]).

## Interpretation

Politeness alone produced no reliable paired change in the three primary
metrics. For harmful scenarios, multi-turn prompting was associated with lower
mean token log-probability, higher perplexity, and higher token entropy. Adding
politeness to multi-turn prompting produced little additional change.

The correlations show associations between observed harmfulness ratings and
distributional metrics. They are not measurements of hidden intent or internal
confidence.

## Limitations

The generated-continuation design does not isolate context from response
content, response length, or stopping behavior. The experiment uses one model,
one seed, and one rater. Paired harmful comparisons contain only 10 scenarios.
The reported p-values are exploratory and are not adjusted for the many related
comparisons. These results do not establish causality and should not be
generalized to other models without replication.

## Privacy and Public-Data Handling

The public tables contain aggregate statistics only. Prompts, model responses,
annotations, record identifiers, token IDs, token strings, token-level
log-probabilities, and private pair rows remain outside the repository.
