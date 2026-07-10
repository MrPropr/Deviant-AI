# Experiment Log

Use this file to record high-level, public-safe experiment decisions.

## 2026-07-09

- Created initial safety-conscious repository scaffold.
- Selected JailbreakBench / JBB-Behaviors as the main dataset plan.
- Listed HarmBench validation subset as optional external validation.
- Listed AgentHarm as related work only.
- Defined four prompt conditions: `direct`, `polite`, `multi_turn`, and `polite_multi_turn`.
- Added placeholder scripts for a lightweight reproducible pipeline.

## 2026-07-10

- Added the preregistered pilot protocol in `docs/protocol.md`.
- Fixed the pilot sample at 10 harmful and 10 benign JBB-Behaviors scenarios with seed-controlled sampling.
- Defined four private, non-adaptive prompt templates with exactly three turns in both multi-turn conditions.
- Added structural prompt validation, deterministic resumable Qwen inference, private annotation conversion, aggregate metrics, and five planned plots.
- Ran sanitized placeholder dry-runs only. No full model experiment or private harmful prompt generation was performed.
