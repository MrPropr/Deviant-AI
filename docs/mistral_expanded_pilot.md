# Expanded Mistral 7B Instruct Pilot

## Scope

This report documents the expanded experiment for
`mistralai/Mistral-7B-Instruct-v0.3`.

The same fixed split and four prompt conditions used for the earlier model
stages were retained. Private prompts, model responses, identifiers, and
record-level annotations are not included in the public repository.

## Configuration

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Quantization: bitsandbytes 4-bit NF4
- Compute dtype: float16
- Decoding: deterministic (`do_sample=false`)
- Seed: 42
- Maximum generation length: 1024 tokens
- Conditions: direct, polite, multi-turn, and polite multi-turn

The experiment included 20 scenarios: 10 harmful and 10 benign. The four
conditions produced 80 conversations and 160 response records. One benign
response reached the fixed generation limit and was retained.

## Annotation

The responses were annotated using the fixed five-level project rubric. The
resulting candidate annotation file passed independent technical validation
and was accepted as the final annotation artifact.

No independent human annotation or additional human reannotation was
performed. This must be considered when interpreting the results.

## Aggregate results

| Condition | Strict ASR | Refusal rate | Partial compliance | Mean harmfulness | Over-refusal | Benign success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 0.20 | 0.30 | 0.50 | 0.575 | 0.00 | 0.90 |
| Polite | 0.20 | 0.30 | 0.50 | 0.600 | 0.00 | 0.90 |
| Multi-turn | 0.30 | 0.10 | 0.60 | 0.750 | 0.00 | 0.90 |
| Polite multi-turn | 0.30 | 0.20 | 0.50 | 0.725 | 0.00 | 0.80 |

## Exploratory paired analysis

Polite wording alone produced no strict-ASR or refusal-rate difference.
Mean harmfulness changed by 0.025
(95% CI [-0.175,
0.250], p =
1.000).

For multi-turn versus direct, strict ASR changed by
0.100 (95% CI
[-0.300, 0.500],
p = 1.000). Mean harmfulness changed by
0.175 (95% CI
[-0.100, 0.450],
p = 0.340), while refusal changed by
-0.200 (95% CI
[-0.600,
0.200], p =
0.625).

Adding politeness to multi-turn prompting changed mean harmfulness by
-0.025 and benign success by
-0.100.

None of the 20 exploratory paired tests reached p < 0.05.
No multiple-comparison correction was applied.

## Interpretation

In this sample, politeness alone changed little. Multi-turn decomposition was
associated with higher mean harmfulness and lower refusal, but the confidence
intervals were wide and included zero in the paired comparisons.

Polite multi-turn prompting did not increase strict ASR relative to ordinary
multi-turn prompting. Its benign success estimate was lower, although this
difference was also not statistically significant.

These results describe one model, one seed, and a small fixed sample. They
should be treated as preliminary rather than as proof of a general effect.

## Figures

- [Strict ASR](../figures/mistral_expanded_asr.png)
- [Mean harmfulness](../figures/mistral_expanded_harmfulness.png)
- [Refusal rate](../figures/mistral_expanded_refusal.png)
- [Benign success](../figures/mistral_expanded_benign_success.png)
- [Multi-turn discovery curve](../figures/mistral_expanded_discovery_curve.png)

## Limitations

- Each condition contains only 10 harmful and 10 benign scenarios.
- The experiment covers one model and one deterministic seed.
- The annotation was not independently reproduced by a human reviewer.
- One benign response reached the 1024-token generation limit.
- Statistical comparisons are exploratory.
- No correction for multiple comparisons was applied.
- Confidence intervals are wide because of the small sample.

## Privacy

Only aggregate tables, figures, configuration, and methodological
documentation may be public. Prompt text, response text, identifiers,
record-level annotations, and judge notes remain private.
