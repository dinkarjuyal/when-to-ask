# when-to-ask

Two small tools for studying whether language models know **when to seek information** and **whether they already know enough**.
Companion code for the blog post *Models solve problems they cannot tell are solvable*.

| folder | what it is | needs |
|---|---|---|
| [`cost_env/`](cost_env/) | A cost-of-information environment with an exactly computable optimal policy, plus the value-of-information rationale distillation that taught a 7B model *when* to investigate, and a transfer test to DiscoveryWorld. | CPU for the environment; one GPU for training |
| [`determinacy_probe/`](determinacy_probe/) | An exact probe of whether a model can tell that an answer is *not yet determined* (QuestBench Logic-Q + a small SAT solver), plus rule-disjoint training data for the judgment. | Any OpenAI-compatible endpoint |

## 1. cost_env: when is information worth its price?

N samples, one contaminated; M instruments, one able to detect it. `TEST sample | instrument` costs points, `GUESS sample`
ends the episode (100 points if correct). The cost per test varies between episodes, so how much to investigate genuinely
depends on it. The optimal policy is computed exactly by dynamic programming, so any policy's **regret** is measurable.
The structure mirrors the *Space Sick* scenario of [DiscoveryWorld](https://github.com/allenai/discoveryworld)
(Jansen et al., 2024), whose own scorer charges nothing for testing.

Result: Qwen2.5-7B learned a calibrated investigate-or-guess policy when trained on rollouts whose written
value-of-information comparison was arithmetically correct *and* matched the action taken (no oracle needed).
Held-out regret fell from 1.92 to 0.046, and the policy transferred to the real DiscoveryWorld scenario with unseen
instruments. See [`cost_env/RESULTS.md`](cost_env/RESULTS.md).

```bash
cd cost_env
python cost_env.py                       # validates the DP optimum against Monte Carlo
bash serve.sh                            # vLLM, Qwen2.5-7B-Instruct, port 8000
python agent.py --split heldout --out base_heldout.json
bash pipeline.sh 2                       # collect -> filter -> QLoRA SFT, two rounds, with evals
```

## 2. determinacy_probe: do models know when they know enough?

QuestBench Logic-Q items are propositional, so whether a goal is determined by the known facts (plus one added fact)
can be computed exactly. The probe asks a model QuestBench's own Yes / No / Not-sure question and measures
**over-claiming**: answering Yes or No when the goal is genuinely undetermined.

Result: Gemini 2.5 Flash solves 91% of fully specified items (98% with thinking) but over-claims 40% of the time
(41% with thinking); Qwen3-8B over-claims 81%. Fine-tuning Qwen3-8B on 4,500 exactly labelled, rule-disjoint
examples raised accuracy when the goal is determined (0.60 to 0.88) but barely reduced over-claiming (0.81 to 0.75).
See [`determinacy_probe/RESULTS.md`](determinacy_probe/RESULTS.md).

```bash
cd determinacy_probe
# QuestBench data: https://storage.googleapis.com/questbench/questbench_data.tar.gz
python make_eval_subset.py --questbench-data ../questbench_data          # the fixed 500-item subset (seed 0)
python logic_solver.py logic_eval500.csv                                   # solver check vs dataset labels
python probe.py --csv logic_eval500.csv --n 100 --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 --no-thinking
python make_training_data.py --questbench-data ../questbench_data          # rule-disjoint training data
```

## Limits

Toy environment; the cost_env success is one 7B model and one DiscoveryWorld scenario (12 seeds per cost).
The determinacy results are on one benchmark (QuestBench Logic-Q), 100 items, one seed per configuration.
The broader set of negative results (curiosity-style fine-tuning and prompting that did not improve question asking)
is described in the blog post.

## Citation and credits

QuestBench: Li et al., 2025 ([arXiv:2503.22674](https://arxiv.org/abs/2503.22674); data from its authors).
DiscoveryWorld: Jansen et al., 2024 ([arXiv:2406.06769](https://arxiv.org/abs/2406.06769); Apache-2.0).
The DiscoveryWorld transfer test (`cost_env/dw_transfer.py`) requires a DiscoveryWorld installation.

MIT licence (this repository's code only).
