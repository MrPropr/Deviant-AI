# Expanded Gemma 2 9B Pilot

## Scope

This report documents the completed expanded pilot for
`google/gemma-2-9b-it`. It replicates the four-condition evaluation previously
run with Qwen, while keeping all prompt and response text private.

## Configuration

- Model: `google/gemma-2-9b-it`
- Quantization: bitsandbytes 4-bit NF4
- Compute dtype: float16
- Decoding: deterministic (`do_sample=false`)
- Seed: 42
- Maximum generation length: 1024 tokens
- Conditions: direct, polite, multi-turn, and polite multi-turn

The experiment included 20 scenarios, split evenly between harmful and benign
labels. The four conditions produced 80 conversations, 160 response records,
and 80 final responses for behavioral analysis.

## Aggregate Results

| Condition | Strict ASR | Refusal Rate | Partial Compliance | Mean Harmfulness | Over-refusal | Benign Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 0.00 | 1.00 | 0.00 | 0.250 | 0.50 | 0.40 |
| Polite | 0.00 | 1.00 | 0.00 | 0.250 | 0.50 | 0.40 |
| Multi-turn | 0.10 | 0.50 | 0.40 | 0.425 | 0.20 | 0.80 |
| Polite multi-turn | 0.10 | 0.50 | 0.40 | 0.425 | 0.20 | 0.80 |

The source for this table is the public aggregate file
`tables/gemma_expanded_metrics.csv`.

## Interpretation

Direct and polite phrasing produced identical aggregate behavioral results, so
politeness alone had no measurable effect in this run. Multi-turn decomposition
was associated with a lower refusal rate, increasing partial compliance, a
strict ASR of 0.10, and higher mean harmfulness. Polite multi-turn prompting did
not differ from ordinary multi-turn prompting.

The same multi-turn conditions also increased benign success from 0.40 to 0.80
and reduced over-refusal from 0.50 to 0.20. These paired changes describe a
safety-utility trade-off in this sample; they do not establish a causal
mechanism.

## Figures

- [Strict ASR](../figures/gemma_expanded_asr.png)
- [Mean harmfulness](../figures/gemma_expanded_harmfulness.png)
- [Refusal rate](../figures/gemma_expanded_refusal.png)
- [Benign success](../figures/gemma_expanded_benign_success.png)

## Limitations

Each condition contains 10 harmful scenarios, so an ASR of 0.10 corresponds to
one scenario. The experiment covers one model, one seed, and one human rater.
An independent second annotation is not yet available. The estimates are
preliminary and should not be generalized to other models or interpreted as
proof of the research hypotheses.

## Privacy

Only aggregate metrics and figures are public. Prompt text, model responses,
scenario identifiers, response identifiers, annotations, and record-level data
remain outside the repository.
