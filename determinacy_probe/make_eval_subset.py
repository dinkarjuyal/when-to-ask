"""Recreate the fixed 500-item Logic-Q evaluation subset used in all reported numbers (random seed 0).

python make_eval_subset.py --questbench-data path/to/questbench_data
(download: https://storage.googleapis.com/questbench/questbench_data.tar.gz)
"""
import argparse
import csv
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--questbench-data", required=True)
ap.add_argument("--out", default="logic_eval500.csv")
a = ap.parse_args()
csv.field_size_limit(10 ** 9)
random.seed(0)
rows = list(csv.DictReader(open(os.path.join(a.questbench_data, "Logic-Q", "simplelogic_heldout_1k.csv"))))
pick = sorted(random.sample(range(len(rows)), 500))
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); [w.writerow(rows[i]) for i in pick]
print(f"{a.out}: 500 of {len(rows)} items")
