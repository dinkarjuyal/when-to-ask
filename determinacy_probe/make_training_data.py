"""Rule-disjoint, solver-labelled determinacy training data (the "train the judgment directly" experiment).

Builds fresh Logic-Q-style items from QuestBench's source rule-set pool (Logic-Q/RP/RP/*.txt), excluding every
rule set used by any of the 1,150 QuestBench Logic-Q items, then labels "is the goal determined if I add fact X?"
questions exactly with logic_solver.py. Output: balanced Yes / No / Not sure chat examples.

python make_training_data.py --questbench-data path/to/questbench_data --per-label 1500 --out determinacy_train.jsonl
"""
import argparse
import ast
import csv
import glob
import json
import os
import random

from logic_solver import determined
from probe import PROMPT, rules_nl

ap = argparse.ArgumentParser()
ap.add_argument("--questbench-data", required=True)
ap.add_argument("--per-label", type=int, default=1500)
ap.add_argument("--rulesets-per-file", type=int, default=120)
ap.add_argument("--out", default="determinacy_train.jsonl")
a = ap.parse_args()
csv.field_size_limit(10 ** 9)
key = lambda cl: frozenset(frozenset(c) for c in cl)
used = {key(ast.literal_eval(r["rules"]))
        for r in csv.DictReader(open(os.path.join(a.questbench_data, "Logic-Q", "simplelogic_heldout_1k.csv")))}
rng = random.Random(1)
pool = []
for f in sorted(glob.glob(os.path.join(a.questbench_data, "Logic-Q", "RP", "RP", "*.txt"))):
    try:
        data = json.loads(open(f).read())
    except Exception:  # one pool file in the release is truncated
        continue
    for rs in rng.sample(data, min(len(data), a.rulesets_per_file)):
        cl = [[concl] + ["not " + p for p in prem] for prem, concl in rs["rules"]]
        if key(cl) not in used:
            pool.append((rs["preds"], cl))
items = []
for preds, cl in pool:
    for _ in range(4):  # QuestBench-style item: goal undetermined, but some single question settles it
        goal = rng.choice(preds); others = [p for p in preds if p != goal]
        facts = rng.sample(others, rng.randint(1, 4)); rest = [p for p in others if p not in facts]
        cannot = rng.sample(rest, min(len(rest), rng.randint(0, 2)))
        row = {"rules": repr(cl), "known_facts": repr(facts), "known_untrue_facts": "[]", "cannot_ask_facts": repr(cannot), "goal": goal}
        if determined(row) != "not sure":
            continue
        cands = sorted({w.split("not ")[-1] for c in cl for w in c} - set(facts) - set(cannot) - {goal})
        if any(determined(row, ["+" + c]) in ("yes", "no") and determined(row, ["-" + c]) in ("yes", "no") for c in cands):
            items.append((row, cands)); break
ex = {"yes": [], "no": [], "not sure": []}
for row, cands in items:
    system = PROMPT.format(rules_nl=rules_nl(ast.literal_eval(row["rules"])))
    base = [f"Alice is {f}." for f in ast.literal_eval(row["known_facts"])]
    pairs = [(None, None)] + [(c, v) for c in cands for v in "+-"]
    for c, v in rng.sample(pairs, min(len(pairs), 10)):
        truth = determined(row, [] if c is None else [v + c])
        if truth not in ex:
            continue
        fact = [] if c is None else [f"Alice is {c}." if v == "+" else f"Alice is not {c}."]
        ex[truth].append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(base + fact) + f"\nIs Alice {row['goal']}?"},
            {"role": "assistant", "content": "Answer: " + {"yes": "Yes", "no": "No", "not sure": "Not sure"}[truth]}]})
n = min(min(len(v) for v in ex.values()), a.per_label)
rows = [x for k in ex for x in rng.sample(ex[k], n)]
rng.shuffle(rows)
with open(a.out, "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
print(f"{len(pool)} fresh rule sets -> {len(items)} items -> {len(rows)} examples ({n} per label) in {a.out}")
