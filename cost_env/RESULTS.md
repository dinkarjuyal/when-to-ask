# cost_env results (Qwen2.5-7B-Instruct, QLoRA)

Regret is the shortfall from the exact optimal expected score (lower is better). Held-out configurations
use sample/instrument counts and costs never seen in training.

| policy | regret, training distribution | regret, held-out |
|---|---|---|
| base model | 1.564 | 1.921 |
| always investigate | 0.481 | 0.643 |
| always guess | 0.229 | 0.118 |
| outcome-ranked expert iteration (learned the base rate, not the cost) | — | 0.174 |
| **VOI rationale distillation, round 1** | 0.180 | 0.076 |
| **VOI rationale distillation, round 2** | **0.098** | **0.046** |

Method: during collection the model writes a short value-of-information comparison in integer points
(guess value, expected tests, expected cost, investigate value, decision). Rollouts are kept only if the arithmetic is
correct and the action follows from the numbers; the template is removed from training prompts. Round 2 collects from
the round-1 model with plain prompts.

## Transfer to DiscoveryWorld *Space Sick* (Easy)

Unseen instruments (densitometer, spectrometer, pH meter, microscope, radiation meter) and item type (mushroom);
12 seeds per cost.

| model | regret at cost 0.03 / 0.10 / 0.25 | task success |
|---|---|---|
| base | 0.84 / 1.42 / 3.23 | 0.36 |
| round 1 | 0.03 / 0 / 0 | 0.69 |
| round 2 | 0.22 / 0 / 0 | 0.64 |

The round-1 vs round-2 gap at cost 0.03 is 1 vs 3 failed episodes out of 12 (Fisher p ≈ 0.6).
