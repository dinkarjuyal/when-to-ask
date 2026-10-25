"""Exact determinacy for QuestBench Logic-Q items.

Each rule list is a clause: ['agreeable', 'not amused', 'not messy'] = agreeable OR NOT amused OR NOT messy.
Known facts / untrue facts are unit clauses. The goal is determined iff exactly one of (goal), (NOT goal) is
entailed, i.e. clauses AND NOT goal is UNSAT, or clauses AND goal is UNSAT. Small DPLL (items have ~20-45 variables).
"""
import ast


def clauses_of(row, extra=()):
    cl = []
    for rule in ast.literal_eval(row["rules"]):
        cl.append(frozenset(("-" + w[4:]) if w.startswith("not ") else ("+" + w) for w in rule))
    for f in ast.literal_eval(row["known_facts"]):
        cl.append(frozenset({"+" + f}))
    for f in ast.literal_eval(row["known_untrue_facts"]):
        cl.append(frozenset({"-" + f}))
    for lit in extra:
        cl.append(frozenset({lit}))
    return cl


def neg(l):
    return ("-" if l[0] == "+" else "+") + l[1:]


def sat(clauses):
    clauses = [set(c) for c in clauses]
    assign = {}
    def simplify(cls, lit):
        out = []
        for c in cls:
            if lit in c:
                continue
            c2 = c - {neg(lit)}
            if not c2:
                return None
            out.append(c2)
        return out
    def dpll(cls):
        while True:
            unit = next((c for c in cls if len(c) == 1), None)
            if unit is None:
                break
            lit = next(iter(unit)); cls = simplify(cls, lit)
            if cls is None:
                return False
        if not cls:
            return True
        lit = next(iter(cls[0]))
        for choice in (lit, neg(lit)):
            s = simplify(cls, choice)
            if s is not None and dpll(s):
                return True
        return False
    return dpll(clauses)


def determined(row, extra=()):
    """Returns 'yes', 'no', or 'not sure' (or 'inconsistent') for the goal given the item plus extra literals."""
    cl = clauses_of(row, extra); g = row["goal"]
    can_true, can_false = sat(cl + [frozenset({"+" + g})]), sat(cl + [frozenset({"-" + g})])
    if can_true and can_false:
        return "not sure"
    if can_true:
        return "yes"
    if can_false:
        return "no"
    return "inconsistent"


if __name__ == "__main__":
    # validation against the dataset's own labels: base item undetermined; every gt question determines the goal both ways
    import csv
    csv.field_size_limit(10 ** 9)
    import sys
    rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "logic_eval500.csv")))[:100]
    base_ok = gt_ok = gt_n = 0
    for r in rows:
        base_ok += determined(r) == "not sure"
        for q in ast.literal_eval(r["gt_qs"]):
            gt_n += 1
            gt_ok += determined(r, ["+" + q]) in ("yes", "no") and determined(r, ["-" + q]) in ("yes", "no")
    print(f"base items undetermined: {base_ok}/100 | gt questions determine both ways: {gt_ok}/{gt_n}")
