"""LLM agent loop for CostEnv: rollouts, scoring, eval metrics.

Each step is a fresh single-turn prompt (rules + explicit state summary),
so a trajectory decomposes into independent (prompt -> response) training
examples. The state summary is the "running hypothesis state" the research
recommends instead of forced reasoning prefixes.

Scoring uses the posterior-EXPECTED return of the actions taken, not the
realized return: a final guess is credited with the posterior probability
it is correct given what was tested. This removes lucky-guess noise (an
immediate guess at low cost is right 1/N of the time, and ranking by
realized return would reinforce exactly that luck).
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time

from openai import OpenAI

from cost_env import CostEnv, optimal_value

SYSTEM = (
    "You are a scientist deciding how much evidence to gather before acting. "
    "Each turn you see the rules and the current state, and choose ONE action. "
    "Think briefly (at most 3 sentences) about whether more testing is worth its cost, "
    "then end with exactly one line in one of these formats:\n"
    "ACTION: TEST <sample> | <instrument>\n"
    "ACTION: GUESS <sample>"
)

TRAIN_CONFIGS = [(2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5)]
TRAIN_COSTS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.25, 0.4]
HELDOUT_CONFIGS = [(2, 6), (5, 3), (3, 6)]
HELDOUT_COSTS = [0.03, 0.1, 0.14, 0.3]


NUMERIC_TEMPLATE = (
    "\n\nBefore choosing, fill in this template with the actual numbers:\n"
    "Guess value = 100/<number of samples> = <value>\n"
    "Expected tests = (<samples> x <instruments> + 1)/2 = <value>\n"
    "Expected test cost = <cost per test> x <expected tests> = <value>\n"
    "Investigate value = 100 - <expected test cost> = <value>\n"
    "Decision: <GUESS or TEST>, because <the larger value> is larger\n"
    "ACTION: GUESS <sample>   (if Decision is GUESS)\n"
    "ACTION: TEST <sample> | <instrument>   (if Decision is TEST)"
)

HINTS = {
    "weak": ("\nRemember: every test costs points, and guessing immediately is always an option. "
             "If a reading is positive, that sample is the contaminated one."),
    "strong": ("\nBefore acting, weigh the options. Guessing right now is correct with probability "
               "1/(number of samples). Finding the contaminated sample by testing usually takes about half "
               "of all sample x instrument pairs, and each test costs the per-test cost. Investigate only if "
               "the expected payoff beats guessing now. Once a reading is positive, guess that sample "
               "immediately. Never repeat a test you already did."),
    # numeric decision procedure for the FIRST move (later moves use "strong"):
    "numeric": ("\nOn your first move, reason with explicit numbers before choosing: "
                "(a) value of guessing now = 100/(number of samples); "
                "(b) expected tests to find the contaminated sample = (samples x instruments + 1)/2; "
                "(c) value of investigating = 100 - cost per test x (b). "
                "State the three numbers, compare (a) and (c), and choose the larger: GUESS now if (a) is "
                "larger, otherwise TEST. While investigating: once a reading is positive, guess that sample "
                "immediately; never repeat a test."),
    # exploration arm for the first move only (STaR-style rationalisation):
    "forceguess": ("\nFor THIS turn you have already decided to guess immediately, without any testing, "
                   "because of the per-test cost. In 1-2 sentences explain the cost-based reason, then "
                   "output ACTION: GUESS <one of the samples>."),
}


def build_user(env: CostEnv) -> str:
    fb = getattr(env, "last_msg", None)
    fb_line = f"\nResult of your last action: {fb}" if fb else ""
    inv = getattr(env, "invalid_actions", 0)
    inv_line = f"\nInvalid actions so far: {inv}" if inv else ""
    return (env.describe_rules() + "\n\nCurrent state:\n" + env.observation() + fb_line + inv_line
            + "\n\nWhat is your next action? Give exactly ONE action.")


def parse_action(text: str):
    """First action only. Models sometimes chain several on one line
    ('ACTION: TEST A | x | ACTION: TEST B | y'); everything after a second
    'ACTION:' (or a newline) is dropped."""
    idx = [m.start() for m in re.finditer(r"ACTION:", text, re.IGNORECASE)]
    if not idx:
        return None
    seg = text[idx[0] + len("ACTION:"):]
    seg = re.split(r"ACTION:|\n", seg, flags=re.IGNORECASE)[0].strip().rstrip("|").strip()
    return seg if re.match(r"(TEST|GUESS)\b", seg, re.IGNORECASE) else None


def _create_with_retry(client, tries=6, **kw):
    for i in range(tries):
        try:
            return client.chat.completions.create(**kw)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def expected_return(env: CostEnv) -> float:
    """Posterior-expected return given the actions taken (see module doc)."""
    cost_paid = env.cost * env.n_tests
    if not env.done:
        return -cost_paid
    pos = [i for (i, k), r in env.tested.items() if "consistent" in r]
    guessed = env._last_guess
    if pos:
        p = 1.0 if guessed == pos[0] else 0.0
    else:
        n_items, n_inst = len(env.items), len(env.instruments)
        untested = [(i, k) for i in range(n_items) for k in range(n_inst) if (i, k) not in env.tested]
        p = sum(1 for (i, _) in untested if i == guessed) / len(untested)
    return p - cost_paid


def run_episode(client, model, n_items, n_inst, cost, seed, temperature, max_tokens=160,
                instruments=None, item_type=None, env=None, hint=None):
    if env is None:
        env = CostEnv(n_items, n_inst, cost, seed, instruments=instruments, item_type=item_type)
    env._last_guess = None
    steps = []
    max_steps = n_items * n_inst + 3
    for _ in range(max_steps):
        user = build_user(env)
        if hint == "numeric" and not steps:
            user += NUMERIC_TEMPLATE
        resp = _create_with_retry(
            client, model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM + (
                          HINTS[hint] if hint and (hint not in ("forceguess", "numeric") or not steps) else
                          (HINTS["strong"] if hint in ("forceguess", "numeric") else ""))},
                      {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        action = parse_action(text)
        steps.append({"user": user, "response": text, "action": action})
        if action is None:
            env.invalid_actions += 1
            env.last_msg = "Your reply had no valid ACTION line. Use 'ACTION: TEST <sample> | <instrument>' or 'ACTION: GUESS <sample>'."
            continue
        if action.upper().startswith("GUESS"):
            g = env._match(action[5:].strip(), env.items)
            env._last_guess = g
        msg, _ = env.step(action)
        env.last_msg = f"{action} -> {msg}" if msg else None
        if env.done:
            break
    first_guess_immediately = bool(steps) and steps[0]["action"] is not None and \
        steps[0]["action"].upper().startswith("GUESS")
    v_opt = optimal_value(n_items, n_inst, cost)
    return {
        "n_items": n_items, "n_inst": n_inst, "cost": cost, "seed": seed,
        "n_tests": env.n_tests, "done": env.done, "invalid": env.invalid_actions,
        "exp_return": expected_return(env), "realized_return": env.episode_return(),
        "v_opt": v_opt, "opt_guess_now": abs(v_opt - 1 / n_items) < 1e-9,
        "guessed_immediately": first_guess_immediately, "steps": steps, "hint": hint,
    }


def summarize(rows, label):
    def agg(rs):
        n = len(rs)
        if n == 0:
            return {}
        return {
            "n": n,
            "exp_return": sum(r["exp_return"] for r in rs) / n,
            "regret": sum(r["v_opt"] - r["exp_return"] for r in rs) / n,
            "mean_tests": sum(r["n_tests"] for r in rs) / n,
            "decision_acc": sum((r["guessed_immediately"] == r["opt_guess_now"]) for r in rs) / n,
            "invalid_rate": sum(r["invalid"] > 0 for r in rs) / n,
            "unfinished": sum(not r["done"] for r in rs) / n,
        }
    out = {"label": label, "all": agg(rows)}
    out["should_investigate"] = agg([r for r in rows if not r["opt_guess_now"]])
    out["should_guess_now"] = agg([r for r in rows if r["opt_guess_now"]])
    by_cost = {}
    for r in rows:
        by_cost.setdefault(r["cost"], []).append(r)
    out["by_cost"] = {str(c): {"mean_tests": sum(x["n_tests"] for x in rs) / len(rs),
                               "guess_now_rate": sum(x["guessed_immediately"] for x in rs) / len(rs),
                               "regret": sum(x["v_opt"] - x["exp_return"] for x in rs) / len(rs)}
                      for c, rs in sorted(by_cost.items())}
    return out


def run_many(jobs, client, model, temperature, workers=64, **kw):
    """jobs: (n, m, cost, seed) or (n, m, cost, seed, hint)."""
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_episode, client, model, *j[:4], temperature,
                          **({**kw, "hint": j[4]} if len(j) > 4 else kw)) for j in jobs]
        for f in cf.as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as e:  # keep going; count failures
                rows.append({"error": str(e)})
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"WARNING: {len(errs)} episodes errored, e.g. {errs[0]['error'][:200]}")
    return [r for r in rows if "error" not in r]


def eval_jobs(split, seeds_per=6, seed0=100000):
    configs, costs = (TRAIN_CONFIGS, TRAIN_COSTS) if split == "train_dist" else (HELDOUT_CONFIGS, HELDOUT_COSTS)
    jobs, s = [], seed0
    for (n, m) in configs:
        for c in costs:
            for _ in range(seeds_per):
                jobs.append((n, m, c, s)); s += 1
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="train_dist", choices=["train_dist", "heldout"])
    ap.add_argument("--seeds-per", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="eval.json")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="unused")
    t0 = time.time()
    seed0 = 100000 if args.split == "train_dist" else 200000
    rows = run_many(eval_jobs(args.split, args.seeds_per, seed0), client, args.model, args.temperature)
    summ = summarize(rows, f"{args.model}|{args.split}|T={args.temperature}")
    summ["elapsed_s"] = time.time() - t0
    print(json.dumps({k: v for k, v in summ.items() if k != "by_cost"}, indent=1))
    print(json.dumps(summ["by_cost"], indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summ, "rows": [{k: v for k, v in r.items() if k != "steps"} for r in rows]}, f)
