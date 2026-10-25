"""Cost-of-information environment family (pure Python, no deps).

Same hidden structure as DiscoveryWorld Space Sick: N items, exactly one
contaminated; M instruments, exactly one "critical" (the only one that can
detect the pathogen). TEST(item, instrument) costs `cost` and returns a
positive reading iff item is contaminated AND instrument is critical.
GUESS(item) ends the episode: +1 if correct, 0 otherwise.

Return = 1[correct guess] - cost * n_tests.

Why this env: the DiscoveryWorld scorer charges nothing for testing, so
"test everything" is weakly dominant and there is no stopping decision to
learn. Here the cost is explicit and varies per episode, so the
reward-optimal amount of investigation genuinely depends on the cost. The
optimal policy is computed exactly by dynamic programming (optimal_value),
so a policy's calibration is measurable as regret against the optimum.

Posterior: (item*, inst*) is uniform over N*M pairs; a negative test
removes exactly that pair, so the posterior stays uniform over untested
pairs. State for the DP is the multiset of untested-pair counts per item.
"""
import random
import re
from functools import lru_cache

TRAIN_INSTRUMENTS = [
    "thermometer", "gas chromatograph", "magnetometer", "Geiger counter",
    "refractometer", "hygrometer", "viscometer", "conductivity probe",
    "UV lamp", "mass spectrometer", "colorimeter", "ultrasound scanner",
    "infrared camera", "oxygen sensor", "salinity meter",
]
TRAIN_ITEM_TYPES = [
    "berry", "root", "leaf", "seed pod", "fern", "lichen", "tuber",
    "fruit", "moss", "flower", "algae", "bark", "nut", "sprout",
]
PATHOGENS = [
    "Bacterium Species 138", "Virus Strain 12", "Fungus Species 3",
    "Parasite Species 7", "Prion Strain 2", "Spore Cluster 9",
]
COSTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35]

# Held out entirely from training so the DiscoveryWorld transfer test uses
# unseen instrument names and an unseen item type.
DW_INSTRUMENTS = ["densitometer", "spectrometer", "PH meter", "microscope", "radiation meter"]


@lru_cache(maxsize=None)
def _V(counts: tuple, cost: float):
    """Optimal expected return from a state where `counts[j]` untested pairs
    remain for item j (sorted desc, zeros dropped). Returns (value, action)
    where action is ('guess', None) or ('test', index_into_counts)."""
    U = sum(counts)
    best_v, best_a = max(counts) / U, ("guess", None)
    for j, cj in enumerate(counts):
        if j > 0 and counts[j] == counts[j - 1]:
            continue  # symmetric items, same value
        nxt = list(counts)
        nxt[j] -= 1
        nxt = tuple(sorted((x for x in nxt if x > 0), reverse=True))
        v_after = _V(nxt, cost)[0] if nxt else 0.0
        v = -cost + (1.0 / U) * 1.0 + ((U - 1) / U) * v_after
        if v > best_v + 1e-12:
            best_v, best_a = v, ("test", j)
    return best_v, best_a


def optimal_value(n_items: int, n_inst: int, cost: float) -> float:
    return _V(tuple([n_inst] * n_items), round(cost, 6))[0]


class CostEnv:
    def __init__(self, n_items, n_inst, cost, seed, instruments=None, item_type=None):
        rng = random.Random(seed)
        self.cost = cost
        self.item_type = item_type or rng.choice(TRAIN_ITEM_TYPES)
        pool = instruments or TRAIN_INSTRUMENTS
        self.instruments = rng.sample(pool, n_inst) if len(pool) > n_inst else list(pool)
        rng.shuffle(self.instruments)
        self.items = [f"{self.item_type} {chr(65 + i)}" for i in range(n_items)]
        self.contaminated = rng.randrange(n_items)
        self.critical = rng.randrange(len(self.instruments))
        self.pathogen = rng.choice(PATHOGENS)
        self.tested = {}  # (item_idx, inst_idx) -> reading
        self.n_tests = 0
        self.done = False
        self.correct = None
        self.invalid_actions = 0

    # ---- interface ----
    def describe_rules(self) -> str:
        return (
            f"Exactly one of {len(self.items)} {self.item_type} samples is contaminated with a pathogen. "
            f"Exactly one of your {len(self.instruments)} instruments can detect it (you don't know which). "
            "Testing a sample with an instrument gives a positive reading only if BOTH the sample is "
            "contaminated AND the instrument is the one that can detect it; otherwise it reads normal.\n"
            f"Every test costs {round(self.cost * 100)} points. Guessing ends the episode: a correct guess earns 100 points, "
            "a wrong guess earns 0. Your score = guess reward minus total test costs. Maximize expected score."
        )

    def observation(self) -> str:
        lines = [f"Samples: {', '.join(self.items)}", f"Instruments: {', '.join(self.instruments)}",
                 f"Cost per test: {round(self.cost * 100)} points",
                 f"Tests used so far: {self.n_tests} (spent {round(self.n_tests * self.cost * 100)} points)"]
        if self.tested:
            lines.append("Test results so far:")
            for (i, k), r in self.tested.items():
                lines.append(f"- {self.items[i]} with {self.instruments[k]}: {r}")
            untested = [(i, k) for i in range(len(self.items)) for k in range(len(self.instruments))
                        if (i, k) not in self.tested]
            lines.append("Untested pairs still available (do not repeat tested pairs):")
            for i in range(len(self.items)):
                rem = [self.instruments[k] for (a, k) in untested if a == i]
                lines.append(f"- {self.items[i]}: {', '.join(rem) if rem else '(all tested)'}")
        else:
            lines.append("No tests done yet.")
        return "\n".join(lines)

    def _match(self, text, names):
        t = re.sub(r"[\s_\-]+", " ", text.lower()).strip()
        if len(t) == 1:  # bare sample letter, e.g. "A"
            hits = [i for i, n in enumerate(names) if n.lower().endswith(" " + t)]
            return hits[0] if len(hits) == 1 else None
        for idx, n in enumerate(names):
            nn = re.sub(r"[\s_\-]+", " ", n.lower()).strip()
            if nn == t:
                return idx
        for idx, n in enumerate(names):
            nn = re.sub(r"[\s_\-]+", " ", n.lower()).strip()
            if nn in t or (len(t) > 2 and t in nn):
                return idx
        return None

    def step(self, action_line: str):
        """action_line: 'TEST <sample> | <instrument>' or 'GUESS <sample>'.
        Returns (observation_text_or_None, done)."""
        m = re.match(r"\s*(TEST|GUESS)\s+(.*)", action_line, re.IGNORECASE)
        if not m:
            self.invalid_actions += 1
            return "Invalid action format.", False
        kind, rest = m.group(1).upper(), m.group(2).strip()
        if kind == "GUESS":
            i = self._match(rest, self.items)
            if i is None:
                self.invalid_actions += 1
                return "Unknown sample.", False
            self.done = True
            self.correct = (i == self.contaminated)
            return None, True
        parts = re.split(r"\s*\|\s*|\s+with\s+|,\s*", rest, maxsplit=1)
        if len(parts) != 2:
            self.invalid_actions += 1
            return "TEST needs a sample and an instrument separated by '|'.", False
        i, k = self._match(parts[0], self.items), self._match(parts[1], self.instruments)
        if i is None or k is None:
            self.invalid_actions += 1
            return "Unknown sample or instrument.", False
        self.n_tests += 1
        if (i, k) in self.tested:
            return f"(already tested) {self.tested[(i, k)]}", False
        reading = ("The readings appear consistent with contamination by " + self.pathogen + "."
                   if (i == self.contaminated and k == self.critical) else "The readings appear normal.")
        self.tested[(i, k)] = reading
        return reading, False

    def episode_return(self) -> float:
        return (1.0 if self.correct else 0.0) - self.cost * self.n_tests

    def optimal(self) -> float:
        return optimal_value(len(self.items), len(self.instruments), self.cost)


def oracle_action(env: CostEnv) -> str:
    """The exact Bayes-optimal next action (used only to validate the DP and
    as a reference ceiling, never as training targets for the main method)."""
    positives = [(i, k) for (i, k), r in env.tested.items() if "consistent" in r]
    if positives:
        return f"GUESS {env.items[positives[0][0]]}"
    untested_by_item = {i: [k for k in range(len(env.instruments)) if (i, k) not in env.tested]
                        for i in range(len(env.items))}
    counts_items = sorted(((len(v), i) for i, v in untested_by_item.items() if v), reverse=True)
    counts = tuple(c for c, _ in counts_items)
    _, (a, j) = _V(counts, round(env.cost, 6))
    if a == "guess":
        return f"GUESS {env.items[counts_items[0][1]]}"
    i = counts_items[j][1]
    return f"TEST {env.items[i]} | {env.instruments[untested_by_item[i][0]]}"


if __name__ == "__main__":
    # Validate DP against Monte Carlo of the oracle policy.
    for (n, m) in [(3, 5), (2, 3), (4, 4)]:
        for c in COSTS:
            rets, tests = [], []
            for s in range(4000):
                env = CostEnv(n, m, c, seed=s)
                while not env.done:
                    env.step(oracle_action(env))
                rets.append(env.episode_return()); tests.append(env.n_tests)
            print(f"N={n} M={m} c={c:<5} DP={optimal_value(n, m, c):.3f} "
                  f"MC={sum(rets)/len(rets):.3f} mean_tests={sum(tests)/len(tests):.2f}")
