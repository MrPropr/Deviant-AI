# Cross-model Behavioral Comparison

## Scope

This report provides a descriptive comparison of the completed expanded
behavioral experiments for:

- `Qwen/Qwen2.5-7B-Instruct`;
- `google/gemma-2-9b-it`;
- `mistralai/Mistral-7B-Instruct-v0.3`.

Only public aggregate behavioral tables are used. Prompt text, response text,
record identifiers, annotation rows, and other private experimental data are
not included.

## Common experimental structure

Each model was evaluated with the same four conditions:

- direct;
- polite;
- multi-turn;
- polite multi-turn.

Each condition contains 10 harmful and 10 benign scenarios. The published
aggregate tables contain no excluded conversations.

## Aggregate behavioral results

| Model | Condition | Strict ASR | Refusal | Partial compliance | Mean harmfulness | Over-refusal | Benign success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 2.5 7B | Direct | 0.10 | 0.90 | 0.00 | 0.300 | 0.00 | 0.90 |
| Qwen 2.5 7B | Polite | 0.10 | 0.90 | 0.00 | 0.325 | 0.00 | 0.90 |
| Qwen 2.5 7B | Multi-turn | 0.70 | 0.20 | 0.10 | 0.825 | 0.00 | 0.90 |
| Qwen 2.5 7B | Polite multi-turn | 0.60 | 0.20 | 0.20 | 0.775 | 0.00 | 0.90 |
| Gemma 2 9B | Direct | 0.00 | 1.00 | 0.00 | 0.250 | 0.50 | 0.40 |
| Gemma 2 9B | Polite | 0.00 | 1.00 | 0.00 | 0.250 | 0.50 | 0.40 |
| Gemma 2 9B | Multi-turn | 0.10 | 0.50 | 0.40 | 0.425 | 0.20 | 0.80 |
| Gemma 2 9B | Polite multi-turn | 0.10 | 0.50 | 0.40 | 0.425 | 0.20 | 0.80 |
| Mistral 7B | Direct | 0.20 | 0.30 | 0.50 | 0.575 | 0.00 | 0.90 |
| Mistral 7B | Polite | 0.20 | 0.30 | 0.50 | 0.600 | 0.00 | 0.90 |
| Mistral 7B | Multi-turn | 0.30 | 0.10 | 0.60 | 0.750 | 0.00 | 0.90 |
| Mistral 7B | Polite multi-turn | 0.30 | 0.20 | 0.50 | 0.725 | 0.00 | 0.80 |

## Descriptive multi-turn effects

The following table reports `multi-turn − direct` differences within each
model.

| Model | Strict ASR Δ | Refusal Δ | Partial compliance Δ | Harmfulness Δ | Benign success Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 2.5 7B | +0.600 | -0.700 | +0.100 | +0.525 | +0.000 |
| Gemma 2 9B | +0.100 | -0.500 | +0.400 | +0.175 | +0.400 |
| Mistral 7B | +0.100 | -0.200 | +0.100 | +0.175 | +0.000 |

Multi-turn prompting increased strict ASR in all three model stages:

- Qwen: +0.600;
- Gemma: +0.100;
- Mistral: +0.100.

Mean harmfulness also increased in all three model stages:

- Qwen: +0.525;
- Gemma: +0.175;
- Mistral: +0.175.

Refusal decreased under multi-turn prompting:

- Qwen: -0.700;
- Gemma: -0.500;
- Mistral: -0.200.

Gemma was the only model stage in which benign success increased under the
multi-turn condition, changing by
+0.400.

## Politeness effects

Polite wording alone did not change strict ASR in any of the three model
stages. Its change in mean harmfulness was small:

- Qwen: +0.025;
- Gemma: +0.000;
- Mistral: +0.025.

Adding politeness to multi-turn prompting did not strengthen strict ASR in
any model stage.

## Interpretation

The most consistent descriptive pattern is associated with multi-turn
decomposition rather than polite wording. All three model stages showed
non-negative changes in strict ASR and mean harmfulness, together with lower
refusal rates.

The largest observed multi-turn shift occurred in the Qwen stage. Gemma and
Mistral showed smaller increases in strict ASR and mean harmfulness.

These values must not be treated as a definitive safety ranking. Annotation
procedures differed between model stages, each condition contains only ten
harmful scenarios, and no unified cross-model inferential analysis is
available.

## Statistical scope

This is a descriptive comparison.

A common cross-model significance test was not calculated because equivalent
published behavioral confidence-interval and paired-statistics tables are not
available for all three model stages. In particular, Gemma does not have the
same published behavioral CI and paired-statistics tables used by Qwen and
Mistral.

Token-probability and fixed-continuation analyses are not included in this
comparison.

## Figures

- [Strict ASR](../figures/cross_model_strict_asr.png)
- [Refusal rate](../figures/cross_model_refusal_rate.png)
- [Partial compliance](../figures/cross_model_partial_compliance.png)
- [Mean harmfulness](../figures/cross_model_harmfulness.png)
- [Benign success](../figures/cross_model_benign_success.png)
- [Multi-turn effects](../figures/cross_model_multi_turn_effects.png)

## Limitations

- Each model-condition cell contains only 10 harmful and 10 benign scenarios.
- Only one deterministic seed was used per model stage.
- Annotation procedures differed across model stages.
- The comparison is based on aggregate behavioral results.
- Cross-model statistical significance was not tested.
- Token-probability results are not included.
- The results should be treated as preliminary.

## Privacy

Only aggregate public metrics, descriptive effects, figures, and
methodological documentation are included. No private prompt, response,
identifier, or record-level annotation data is used.
