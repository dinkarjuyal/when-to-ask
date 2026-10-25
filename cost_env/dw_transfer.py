"""Transfer test: run the SAME agent loop (same prompt, same action format,
same posterior-expected scoring) on the real DiscoveryWorld Space Sick
simulator. Instrument names (densitometer, spectrometer, PH meter,
microscope, radiation meter) and the item type (mushroom) were held out of
synthetic training entirely, and readings are the simulator's real text.

DWEnv mirrors CostEnv's interface; each TEST/GUESS is executed through the
DiscoveryWorld action API (teleport/pickup/use/put + tick). The DW scorer's
own success flag is recorded for sanity, but the reported metric is the
cost-aware expected return, since DW's raw scorecard charges nothing for
testing.

Usage (local, DW venv, vLLM reachable at --base-url):
  ../discoveryworld/.venv/bin/python dw_transfer.py --model curious --costs 0.02,0.05,0.12,0.25 --seeds 0-9
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from openai import OpenAI

from cost_env import CostEnv, optimal_value
from agent import run_episode, summarize

DEVICE_KEYS = ("densitometer", "spectrometer", "ph meter", "microscope", "radiation meter")


def _nearby(obs, keys):
    out, seen = [], set()
    for objs in obs["ui"].get("nearbyObjects", {}).get("objects", {}).values():
        for o in objs:
            n = o.get("name", "").lower()
            if any(k in n for k in keys) and o["uuid"] not in seen:
                out.append(o); seen.add(o["uuid"])
    return out


class DWEnv(CostEnv):
    def __init__(self, cost, seed, thread_id):
        self.cost = cost
        self.api = DiscoveryWorldAPI(threadID=thread_id)
        assert self.api.loadScenario(scenarioName="Space Sick", difficultyStr="Easy",
                                     randomSeed=seed, numUserAgents=1)
        obs = self.api.getAgentObservation(agentIdx=0)
        foods = _nearby(obs, ("mushroom",))
        insts = _nearby(obs, DEVICE_KEYS)
        jars = _nearby(obs, ("jar",))
        assert len(foods) == 3 and len(insts) == 5 and jars, "objects not all visible"
        rng = random.Random(seed)
        rng.shuffle(foods); rng.shuffle(insts)
        self.item_type = "mushroom"
        self.items = [f"mushroom {chr(65 + i)}" for i in range(len(foods))]
        self.instruments = [o["name"] for o in insts]
        self._food_uuid = [o["uuid"] for o in foods]
        self._inst_uuid = [o["uuid"] for o in insts]
        self._jar = jars[0]["uuid"]
        info = self.api.world.taskScorer.tasks[0].scoringInfo
        self.contaminated = self._food_uuid.index(info["objectToMove"].uuid)  # scoring only
        self.critical = self._inst_uuid.index(info["criticalInstrument"].uuid)  # scoring only
        self.pathogen = "the pathogen"
        self.tested, self.n_tests, self.done, self.correct, self.invalid_actions = {}, 0, False, None, 0
        self._held = set()
        self.dw_success = None

    def _act(self, a, x=None, y=None):
        self.api.performAgentAction(agentIdx=0, actionJSON={"action": a, "arg1": x, "arg2": y})
        self.api.tick()
        return self.api.getAgentObservation(agentIdx=0)["ui"]["lastActionMessage"]

    def _hold(self, i):
        if i not in self._held:
            self._act("TELEPORT_TO_OBJECT", self._food_uuid[i]); self._act("PICKUP", self._food_uuid[i])
            self._held.add(i)

    def step(self, action_line):
        import re
        m = re.match(r"\s*(TEST|GUESS)\s+(.*)", action_line, re.IGNORECASE)
        if not m:
            self.invalid_actions += 1; return "Invalid action format.", False
        kind, rest = m.group(1).upper(), m.group(2).strip()
        if kind == "GUESS":
            i = self._match(rest, self.items)
            if i is None:
                self.invalid_actions += 1; return "Unknown sample.", False
            self._hold(i)
            self._act("TELEPORT_TO_OBJECT", self._jar); self._act("PUT", self._food_uuid[i], self._jar)
            self.dw_success = self.api.getTaskScorecard()[0]["completedSuccessfully"]
            self.done, self.correct = True, (i == self.contaminated)
            return None, True
        parts = re.split(r"\s*\|\s*|\s+with\s+|,\s*", rest, maxsplit=1)
        if len(parts) != 2:
            self.invalid_actions += 1; return "TEST needs sample | instrument.", False
        i, k = self._match(parts[0], self.items), self._match(parts[1], self.instruments)
        if i is None or k is None:
            self.invalid_actions += 1; return "Unknown sample or instrument.", False
        self.n_tests += 1
        if (i, k) in self.tested:
            return self.tested[(i, k)], False
        self._hold(i)
        self._act("TELEPORT_TO_OBJECT", self._inst_uuid[k])
        msg = self._act("USE", self._inst_uuid[k], self._food_uuid[i])
        reading = msg.strip().split("\n")[-1].strip() or msg.strip()
        # normalise DW's phrasing to the shared "consistent with" / "normal" vocabulary check
        self.tested[(i, k)] = reading
        return reading, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--costs", default="0.02,0.05,0.12,0.25")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="dw_transfer.json")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="unused")
    a, b = map(int, args.seeds.split("-"))
    rows = []
    tid = 30000 + (os.getpid() % 1000) * 100
    for c in [float(x) for x in args.costs.split(",")]:
        for s in range(a, b + 1):
            tid += 1
            env = DWEnv(c, s, tid)
            r = run_episode(client, args.model, 3, 5, c, s, args.temperature, env=env)
            r["dw_success"] = env.dw_success
            r.pop("steps", None)
            rows.append(r)
            print(f"c={c} seed={s} tests={r['n_tests']} guess_now={r['guessed_immediately']} "
                  f"opt_guess_now={r['opt_guess_now']} exp_ret={r['exp_return']:.3f} v_opt={r['v_opt']:.3f} "
                  f"dw_success={env.dw_success}", flush=True)
    summ = summarize(rows, f"DW|{args.model}")
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
