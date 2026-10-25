"""Does a model recognise when an answer is NOT yet determined by what it knows?

QuestBench Logic-Q items are propositional, so logic_solver.py gives exact ground truth for
"is the goal determined if I add fact X?". For each item we take 2 (candidate, value) pairs that truly
determine the goal and 2 that leave it undetermined, ask the model QuestBench's own Yes/No/Not-sure
prompt with that fact added, and report:

  over_claim  = P(answers Yes/No | goal truly undetermined)   -- the failure that matters
  under_claim = P(answers "Not sure" | goal truly determined)
  acc_when_determined = P(correct Yes/No | goal truly determined)

Works with any OpenAI-compatible endpoint (vLLM, OpenAI, Gemini's OpenAI-compatible API, ...).

Example:
  python probe.py --csv logic_eval500.csv --n 100 --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B
  python probe.py --csv logic_eval500.csv --n 100 --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
                  --model gemini-2.5-flash --api-key $GEMINI_API_KEY
"""
import argparse
import ast
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from logic_solver import determined

PROMPT = """Suppose you know the following rules about Alice:
{rules_nl}

You will presented with a binary question about an attribute of Alice. Please answer it with "Yes" or "No" or "Not sure".
Reason step-by-step, then generate "Answer:" followed by the answer and nothing else."""  # QuestBench's own isambig CoT prompt


def rules_nl(rules):
    out = []
    for rule in rules:
        neg = [w.split("not ")[-1] for w in rule if w.startswith("not ")]
        pos = [w for w in rule if not w.startswith("not ")]
        out.append(f"If Alice is {' and '.join(neg)}, then Alice is {pos[0]}.")
    return "\n".join(out)


def candidates(row):
    """Askable attributes derived only from what the prompt shows."""
    rules = ast.literal_eval(row["rules"])
    known = set(ast.literal_eval(row["known_facts"])) | set(ast.literal_eval(row["known_untrue_facts"]))
    cannot = set(ast.literal_eval(row["cannot_ask_facts"]))
    return sorted({w.split("not ")[-1] for rule in rules for w in rule} - known - cannot - {row["goal"]})


def probes_for(row, rng):
    det, und = [], []
    for c in candidates(row):
        for v in ("+", "-"):
            truth = determined(row, [v + c])
            (det if truth in ("yes", "no") else und if truth == "not sure" else []).append((c, v, truth))
    rng.shuffle(det); rng.shuffle(und)
    return det[:2] + und[:2]


def parse(text):
    a = (text or "").split("Answer:")[-1].lower()
    return "not sure" if "not sure" in a else ("yes" if "yes" in a else ("no" if "no" in a else "unparsed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="QuestBench Logic-Q csv (e.g. the 500-item subset from make_eval_subset.py)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="unused")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False (Qwen3-style templates)")
    ap.add_argument("--out", default="probe_result.json")
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key=a.api_key)
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(a.csv)))[:a.n]
    rng = random.Random(0)
    jobs = []
    for r in rows:
        system = PROMPT.format(rules_nl=rules_nl(ast.literal_eval(r["rules"])))
        base = [f"Alice is {f}." for f in ast.literal_eval(r["known_facts"])] + \
               [f"Alice is not {f}." for f in ast.literal_eval(r["known_untrue_facts"])]
        for c, v, truth in probes_for(r, rng):
            fact = f"Alice is {c}." if v == "+" else f"Alice is not {c}."
            jobs.append((system, "\n".join(base + [fact]) + f"\nIs Alice {r['goal']}?", truth))
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else None

    def ask(j):
        r = client.chat.completions.create(model=a.model, temperature=0.0, max_tokens=a.max_tokens, extra_body=extra,
                                           messages=[{"role": "system", "content": j[0]}, {"role": "user", "content": j[1]}])
        return parse(r.choices[0].message.content)

    with ThreadPoolExecutor(16) as ex:
        ans = list(ex.map(ask, jobs))
    und = [x for x, j in zip(ans, jobs) if j[2] == "not sure"]
    det = [(x, j[2]) for x, j in zip(ans, jobs) if j[2] in ("yes", "no")]
    res = {"model": a.model, "n_items": len(rows),
           "over_claim": round(sum(x != "not sure" for x in und) / max(1, len(und)), 3),
           "under_claim": round(sum(x == "not sure" for x, _ in det) / max(1, len(det)), 3),
           "acc_when_determined": round(sum(x == t for x, t in det) / max(1, len(det)), 3),
           "n_undetermined": len(und), "n_determined": len(det)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
