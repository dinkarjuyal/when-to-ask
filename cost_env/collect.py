"""Expert-iteration data collection: sample K episodes per training config
from the current policy, keep the best by posterior-EXPECTED return when it
clearly beats its group mean (positive advantage), and emit its per-step
(prompt -> response) pairs as SFT examples.

This is on-policy and outcome-ranked (ReST / expert iteration), not
imitation of curious-looking behavior -- an episode is kept because its
decisions scored best net of test costs, whatever they looked like.
"""
import argparse
import json
import random
import re
import time

from openai import OpenAI

from agent import NUMERIC_TEMPLATE, SYSTEM, TRAIN_CONFIGS, TRAIN_COSTS, run_many
from cost_env import optimal_value


def numeric_ok(r):
    """First-move rationale check, arithmetic only (no oracle): numbers match the stated
    formulas for this episode's N, M, cost, and the decision follows from them, and the
    action taken matches the decision."""
    t = r["steps"][0]["response"]
    n, m, c = r["n_items"], r["n_inst"], r["cost"]
    g = re.search(r"Guess value\s*=[^=\n]*=\s*(-?[0-9]*\.?[0-9]+)", t)
    v = re.search(r"Investigate value\s*=[^=\n]*=\s*(-?[0-9]*\.?[0-9]+)", t)
    d = re.search(r"Decision:\s*(GUESS|TEST)", t, re.I)
    if not (g and v and d):
        return False
    gv, iv, dec = float(g.group(1)), float(v.group(1)), d.group(1).upper()
    if abs(gv - 100 / n) > 1.0 or abs(iv - (100 - round(c * 100) * (n * m + 1) / 2)) > 1.5:
        return False
    if dec != ("GUESS" if gv > iv else "TEST"):
        return False
    act = r["steps"][0]["action"] or ""
    if dec == "TEST" and not re.match(r"TEST\s+.+(\||\bwith\b|,).+", act, re.I):
        return False  # first action must itself be a valid TEST (sample AND instrument)
    return (dec == "GUESS") == r["guessed_immediately"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--groups-per-cell", type=int, default=6, help="seeds per (config, cost) cell")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-adv", type=float, default=0.02)
    ap.add_argument("--hint-mix", default="plain:2,weak:1,strong:3,forceguess:2",
                    help="exploration prompts per group; training targets always use the plain prompt")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--require-consistent", action="store_true",
                    help="keep only episodes whose first-move numeric rationale is arithmetically correct "
                         "and whose decision follows from its own numbers (oracle-free check)")
    ap.add_argument("--balance", action="store_true", help="equalise guess vs test first-move examples")
    ap.add_argument("--out", default="sft_round.jsonl")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="unused")
    mix = [(x.split(":")[0], int(x.split(":")[1])) for x in args.hint_mix.split(",")]
    jobs, groups, s = [], [], args.seed0
    for (n, m) in TRAIN_CONFIGS:
        for c in TRAIN_COSTS:
            for _ in range(args.groups_per_cell):
                groups.append((n, m, c, s))
                for name, cnt in mix:
                    jobs.extend([(n, m, c, s, None if name == "plain" else name)] * cnt)
                s += 1
    t0 = time.time()
    rows = run_many(jobs, client, args.model, args.temperature, workers=96)
    print(f"{len(rows)} episodes in {time.time()-t0:.0f}s")

    by_group = {}
    for r in rows:
        by_group.setdefault((r["n_items"], r["n_inst"], r["cost"], r["seed"]), []).append(r)

    kept, n_examples, stats = [], 0, {"groups": 0, "kept": 0, "degenerate": 0}
    with open(args.out, "w") as f:
        # Decide guess-now vs investigate per (config, cost) CELL: compare the exact
        # guess value 1/N against the mean over the cell's groups of each group's
        # best investigating episode. Averaging best-of-group across groups cancels
        # lucky early hits while still reflecting competent investigation.
        cell_best_inv = {}
        for key, rs in by_group.items():
            inv = [r["exp_return"] for r in rs if not r["guessed_immediately"] and r["done"] and r["invalid"] == 0]
            cell_best_inv.setdefault(key[:3], []).append(max(inv) if inv else -1.0)
        cell_choice = {cell: ("guess" if 1.0 / cell[0] > sum(v) / len(v) else "investigate")
                       for cell, v in cell_best_inv.items()}
        stats["cells_choose_guess"] = sum(1 for v in cell_choice.values() if v == "guess")
        for key, rs in by_group.items():
            stats["groups"] += 1
            v_guess = 1.0 / key[0]
            if args.require_consistent:
                # decision is fixed by the (arithmetic-checked) rationale; returns judge the rest
                cand = [r for r in rs if r["done"] and r["invalid"] == 0 and numeric_ok(r)]
                if not cand:
                    stats["degenerate"] += 1; continue
                if cand[0]["guessed_immediately"]:
                    best = cand[0]
                else:
                    best = max(cand, key=lambda r: (r["exp_return"], -r["n_tests"]))
                    if best["exp_return"] <= v_guess:
                        stats["degenerate"] += 1; continue
                stats["kept"] += 1
                kept.append(best)
                continue
            if cell_choice[key[:3]] == "guess":
                g = [r for r in rs if r["guessed_immediately"] and r["done"] and r["invalid"] == 0
                     and not re.search(r"no cost|free|costs? nothing|zero cost", r["steps"][0]["response"], re.I)
                     and (not args.require_consistent or numeric_ok(r))]
                if not g:
                    stats["degenerate"] += 1; continue
                best = g[0]
            else:
                ok = [r for r in rs if not r["guessed_immediately"] and r["done"] and r["invalid"] == 0
                      and (not args.require_consistent or numeric_ok(r))]
                if not ok:
                    stats["degenerate"] += 1; continue
                best = max(ok, key=lambda r: (r["exp_return"], -r["n_tests"]))
                mean = sum(r["exp_return"] for r in rs) / len(rs)
                if best["exp_return"] <= v_guess or best["exp_return"] - mean < args.min_adv:
                    stats["degenerate"] += 1; continue
            stats["kept"] += 1
            kept.append(best)
        if args.balance and kept:
            gk = [r for r in kept if r["guessed_immediately"]]
            tk = [r for r in kept if not r["guessed_immediately"]]
            k = min(len(gk), len(tk))
            rnd = random.Random(0)
            kept = rnd.sample(gk, k) + rnd.sample(tk, k)
            stats["balanced_to_each"] = k
        for best in kept:
            for st in best["steps"]:
                if st["action"] is None:
                    continue
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": st["user"].replace(NUMERIC_TEMPLATE, "")},
                    {"role": "assistant", "content": st["response"]},
                ], "cost": best["cost"], "exp_return": best["exp_return"], "v_opt": best["v_opt"],
                   "hint": best["hint"]}) + "\n")
                n_examples += 1

    # sampling-policy quality this round (mean over all samples, not just kept)
    n = len(rows)
    print(json.dumps({**stats, "sft_examples": n_examples,
                      "policy_exp_return": sum(r["exp_return"] for r in rows) / n,
                      "policy_regret": sum(r["v_opt"] - r["exp_return"] for r in rows) / n,
                      "kept_regret": (sum(r["v_opt"] - r["exp_return"] for r in kept) / len(kept)) if kept else None,
                      "kept_guess_now_rate": (sum(r["guessed_immediately"] for r in kept) / len(kept)) if kept else None,
                      "kept_by_hint": {h: sum(1 for r in kept if (r["hint"] or "plain") == h) for h in ["plain", "weak", "strong", "forceguess"]},
                      "regret_by_hint": {h: (sum(r["v_opt"] - r["exp_return"] for r in rows if (r["hint"] or "plain") == h) /
                                             max(1, sum(1 for r in rows if (r["hint"] or "plain") == h))) for h in ["plain", "weak", "strong", "forceguess"]},
                      "should_guess_now_cells_kept": sum(1 for r in kept if r["opt_guess_now"]),
                      "kept_decision_matches_opt": (sum((r["guessed_immediately"] == r["opt_guess_now"]) for r in kept) / len(kept)) if kept else None,
                      "should_guess_now_cells_total": sum(1 for g in groups if abs(optimal_value(g[0], g[1], g[2]) - 1 / g[0]) < 1e-9)},
                     indent=1))


if __name__ == "__main__":
    main()
