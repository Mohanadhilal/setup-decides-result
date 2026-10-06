#!/usr/bin/env python3
"""
gate_sweep_all.py
=================
Re-checks the first paragraph of Section 5.7 on the final protocol. Runs the
gate sweep of gate_sweep_protocol.py for preferences 0, 1, 2 and 4, merges it
with the preference-3 file you already have, and prints the three numbers the
paragraph states:

  * how many preferences keep the same checkpoint under all four rules
  * the width of the front in diffuser steam under each rule
  * Kruskal-Wallis p across the five preferences under each rule

Needs gate_sweep_protocol.py and gate_sweep_protocol_results.csv (pref 3) in
the same folder. Run from E:\\SugerExtraction with the venv active:

    python gate_sweep_all.py --workers 5
"""
import os, csv, argparse
import numpy as np
from multiprocessing import Pool
from scipy import stats

import train_multiseed as tm
import gate_sweep_protocol as G

RULES = list(G.RULES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", nargs="*", type=int, default=[0, 1, 2, 4])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    jobs = [("SAC", p, s) for p in a.prefs for s in tm.TRAIN_SEEDS]
    with Pool(a.workers) as pool:
        res = [r for rr in pool.map(G.process_run, jobs) for r in rr]
    if os.path.exists("gate_sweep_protocol_results.csv"):          # pref 3 from the earlier run
        for r in csv.DictReader(open("gate_sweep_protocol_results.csv")):
            if int(r["pref_i"]) not in a.prefs:
                for k in ("Gp", "T12", "safe_seed_frac_test"): r[k] = float(r[k])
                r["pref_i"] = int(r["pref_i"]); r["seed"] = int(r["seed"]); r["ckpt"] = int(r["ckpt"])
                res.append(r)
    tm.write_csv("gate_sweep_all_results.csv", res)
    prefs = sorted({int(r["pref_i"]) for r in res})

    print("\nCHECKPOINT CHANGES against the rule used (runs out of 5)")
    print(f"{'pref':6s}" + "".join(f"{r:>9}" for r in RULES[1:]) + "   same under all rules")
    same_all = 0
    for p in prefs:
        strict = {int(r["seed"]): int(r["ckpt"]) for r in res if int(r["pref_i"]) == p and r["rule"] == "strict"}
        ch = {rule: sum(1 for r in res if int(r["pref_i"]) == p and r["rule"] == rule and int(r["ckpt"]) != strict[int(r["seed"])])
              for rule in RULES[1:]}
        same = all(v == 0 for v in ch.values()); same_all += same
        print(f"{p:<6d}" + "".join(f"{ch[r]:>9d}" for r in RULES[1:]) + f"   {'yes' if same else 'no'}")
    print(f"preferences with the same checkpoint under all four rules: {same_all} of {len(prefs)}")

    print("\nFRONT AND DISTINCTNESS under each rule (five preferences, five runs each)")
    print(f"{'rule':8s}{'Gp width':>10}{'T12 width':>11}{'KW p, Gp':>11}{'KW p, T12':>11}")
    for rule in RULES:
        gp = [[float(r["Gp"]) for r in res if int(r["pref_i"]) == p and r["rule"] == rule] for p in prefs]
        tt = [[float(r["T12"]) for r in res if int(r["pref_i"]) == p and r["rule"] == rule] for p in prefs]
        wg = max(map(np.mean, gp)) - min(map(np.mean, gp)); wt = max(map(np.mean, tt)) - min(map(np.mean, tt))
        print(f"{rule:8s}{wg:>10.3f}{wt:>11.2f}{stats.kruskal(*gp).pvalue:>11.4f}{stats.kruskal(*tt).pvalue:>11.4f}")
    print("\nwrote gate_sweep_all_results.csv")


if __name__ == "__main__":
    main()
