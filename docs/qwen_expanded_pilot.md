# Expanded Qwen Pilot

## Scope

This report documents the first completed expanded model stage in Deviant-AI, using `Qwen/Qwen2.5-7B-Instruct`. Gemma and Mistral expanded stages were completed later and are documented separately.

## Model and Configuration

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: bitsandbytes 4-bit NF4
- Compute dtype: `float16`
- Decoding: deterministic (`do_sample=false`)
- Seed: 42
- Maximum generation length: 1024 tokens per response
- Maximum multi-turn length: 3 turns

The public configuration is stored in `configs/qwen_expanded_pilot.yaml`.

## Dataset Split

The private experimental split contained 20 scenarios: 10 harmful and 10 benign. Four prompt conditions produced 80 conversations and 160 individual response records. No prompt or response text is included in the public repository.

## Prompt Conditions

The four conditions were `direct`, `polite`, `multi_turn`, and `polite_multi_turn`. Direct and polite conversations used one turn. Both multi-turn conditions used exactly three turns.

## Token-Budget Policy

The fixed generation budget was 1024 tokens per response. Twenty responses reached that limit. They were retained because the same fixed budget was part of every condition, the available text was sufficient for manual scoring, and reaching the limit was not treated as a generation error. No response was excluded for this reason.

## Annotation and Second Review

Human annotation used the public five-level scoring rubric. A second review was performed on 38 flagged rows: 36 scores were confirmed, 2 were revised, and 0 rows were excluded. This was a review of flagged rows, not independent annotation by a second reviewer and not an inter-annotator agreement study.

## Aggregate Results

| Condition | Strict ASR | Refusal Rate | Partial Compliance | Mean Harmfulness | Benign Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 0.10 | 0.90 | 0.00 | 0.300 | 0.90 |
| Polite | 0.10 | 0.90 | 0.00 | 0.325 | 0.90 |
| Multi-turn | 0.70 | 0.20 | 0.10 | 0.825 | 0.90 |
| Polite multi-turn | 0.60 | 0.20 | 0.20 | 0.775 | 0.90 |

The analysis groups records by `model_id`, `scenario_id`, and `condition`. For each harmful conversation, it uses the maximum harmfulness score across turns. Strict ASR is the rate with score `1.00`; refusal is the rate with score `0.00` or `0.25`; partial compliance is the rate with score `0.50` or `0.75`; and first harmful turn is the first turn with score at least `0.75`.

## Exploratory Paired Analysis

For multi-turn versus direct, strict ASR increased by 0.600 (95% confidence interval [0.300, 0.900], p = 0.031), mean harmfulness increased by 0.525 (95% confidence interval [0.300, 0.725], p = 0.016), and refusal rate decreased by 0.700 (95% confidence interval [-1.000, -0.400], p = 0.016).

For polite versus direct, the strict ASR difference was 0.000, the mean harmfulness difference was 0.025, and the refusal-rate difference was 0.000. For polite multi-turn versus multi-turn, the strict ASR difference was -0.100 (p = 1.000) and the mean harmfulness difference was -0.050 (p = 0.500).

These paired results are exploratory. Confidence interval is abbreviated as CI in the public tables and figures.

## Main Observations

Within this small sample, polite wording alone changed little. Multi-turn decomposition was associated with higher strict ASR and mean harmfulness and with lower refusal. Adding polite wording to the multi-turn condition did not strengthen those effects. Benign success remained 0.90 in every condition. These results are preliminary evidence from the Qwen stage; the later cross-model report compares the completed behavioral stages descriptively without treating them as a unified confirmatory test.

## Limitations

- This report covers one model; the later cross-model comparison is descriptive rather than inferential.
- The sample contains 10 harmful scenarios and 10 benign scenarios.
- Annotation was manual.
- The second review covered flagged rows and was not performed by an independent reviewer.
- Twenty responses reached the fixed token limit.
- Statistical results are exploratory.
- No correction for multiple comparisons has been applied yet.

## Public and Private Artifacts

Public artifacts include aggregate metrics, confidence intervals, exploratory paired statistics, interaction effects, graph values, and four figures under `tables/` and `figures/`. The figures can be reproduced with:

```bash
python -m src.plot_expanded_results \
  --metrics tables/qwen_expanded_metrics.csv \
  --graph-values tables/qwen_expanded_graph_values.csv \
  --output-dir figures
```

Private prompts, raw outputs, response text, annotation tables, and second-review logs remain only in ignored local directories such as `data/private/`, `prompts/private/`, `results/raw/`, and `results/judged/`.
