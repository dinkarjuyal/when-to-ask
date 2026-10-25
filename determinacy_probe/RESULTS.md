# determinacy_probe results (QuestBench Logic-Q, first 100 items of the fixed 500-item subset)

Each item contributes 2 (fact, value) probes that truly determine the goal and 2 that leave it undetermined
(exact labels from `logic_solver.py`; ~190 undetermined and ~200 determined probes per model).

| model | over-claims (Yes/No when undetermined) | under-claims ("Not sure" when determined) | correct when determined |
|---|---|---|---|
| Gemini 2.5 Flash, thinking off | 0.40 | 0.32 | 0.63 |
| Gemini 2.5 Flash, thinking 2,048 tokens | 0.41 | 0.27 | 0.71 |
| Qwen3-8B, thinking off | 0.81 | 0.16 | 0.60 |
| Qwen3-8B + determinacy SFT (4,500 rule-disjoint examples) | 0.75 | 0.10 | 0.88 |

For reference, Gemini 2.5 Flash on the *fully specified* version of the same items (QuestBench full-information mode):
0.91 (thinking off), 0.98 (thinking on). On the original underspecified task (pick the one question that resolves the
item) it scores 0.295.

Notes
- The training data comes from QuestBench's source rule-set pool, excluding every rule set used by any of the 1,150
  Logic-Q items (`make_training_data.py`). An earlier version reused evaluation rule sets (592 of 643 items) and was discarded.
- After determinacy SFT, Qwen3-8B stopped producing QuestBench's "Question: Is Alice X?" format and scored 0.03 on the
  original task (base 0.156). This is a format effect from narrow training, not a capability result.
- Gemini runs used the Vertex AI API (temperature 0); Qwen runs used vLLM with thinking disabled.
