#!/usr/bin/env python3
"""
table7_rebuild.py
=================
Editor's point 2: Table 7 gave "29 of 300" seeds leaving the window, but the
29 were counted under the quality-drop scenario alone while the denominator
covered all three scenarios. If the combined scenario reproduces the quality
scenario, it contributes its own violations; with the single pressure-sag excursion of
0.012 degrees the pooled count is 59 (29 quality, 29 combined, 1 pressure sag).

This script recomputes every Table 7 cell from robustness_perseed.csv with the
pooling stated explicitly, and prints the per-scenario breakdown the text
needs. Run from the project folder:

    python table7_rebuild.py                      # reads runs_multiseed/robustness_perseed.csv
    python table7_rebuild.py --file path/to/robustness_perseed.csv
"""
import csv, argparse
from collections import defaultdict
import numpy as np

SCEN = ["quality", "pressure", "combined"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="runs_multiseed/robustness_perseed.csv")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.file)))
    for r in rows:
        for k in ("peak_excursion", "cc_drop"): r[k] = float(r[k])
    ctrls = sorted({r["controller"] for r in rows}, key=lambda c: (c.startswith("MORL"), c))
    learned = [c for c in ctrls if c.startswith("MORL")]

    def agg(sel, label):
        pk = np.array([r["peak_excursion"] for r in sel]); cc = np.array([r["cc_drop"] for r in sel])
        left = int(np.sum(pk > 0.0))
        return dict(controller=label, n=len(sel), peak_mean=round(pk.mean(), 3),
                    peak_P95=round(float(np.percentile(pk, 95)), 3), peak_worst=round(pk.max(), 3),
                    left=left, cc_mean=round(cc.mean(), 3), cc_worst=round(cc.max(), 3))

    groups = [("PID", lambda r: r["controller"] == "PID"),
              ("MPC", lambda r: r["controller"] == "MPC"),
              ("Fuzzy", lambda r: r["controller"] == "Fuzzy"),
              ("MORL-SAC, all runs", lambda r: r["controller"].startswith("MORL")),
              ("MORL-SAC, run 0", lambda r: r["controller"] == "MORL-SAC run0"),
              ("MORL-SAC, run 1", lambda r: r["controller"] == "MORL-SAC run1"),
              ("MORL-SAC, runs 2 to 4", lambda r: r["controller"] in ("MORL-SAC run2", "MORL-SAC run3", "MORL-SAC run4"))]

    print("TABLE 7, pooled over the three scenarios (what the table reports)")
    print(f"{'controller':24s}{'n':>5}{'peak mean':>11}{'P95':>8}{'worst':>8}{'left window':>14}{'Cc mean':>9}{'Cc worst':>10}")
    out = []
    for label, f in groups:
        sel = [r for r in rows if f(r)]
        if not sel: continue
        g = agg(sel, label); out.append(g)
        print(f"{label:24s}{g['n']:>5}{g['peak_mean']:>11.3f}{g['peak_P95']:>8.3f}{g['peak_worst']:>8.3f}"
              f"{str(g['left'])+' of '+str(g['n']):>14}{g['cc_mean']:>9.3f}{g['cc_worst']:>10.3f}")
    with open("table7_rebuilt.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    by_scen = []
    print("\nPER-SCENARIO BREAKDOWN, learned controller (for the text of Section 5.5)")
    print(f"{'scenario':10s}{'n':>5}{'left window':>13}{'peak mean':>11}{'P95':>8}{'worst':>8}{'Cc mean':>9}")
    for s in SCEN:
        sel = [r for r in rows if r["scenario"] == s and r["controller"].startswith("MORL")]
        g = agg(sel, s); by_scen.append(dict(scenario=s, **{k: v for k, v in g.items() if k != "controller"}))
        print(f"{s:10s}{g['n']:>5}{str(g['left'])+' of '+str(g['n']):>13}{g['peak_mean']:>11.3f}{g['peak_P95']:>8.3f}{g['peak_worst']:>8.3f}{g['cc_mean']:>9.3f}")
    with open("table7_by_scenario.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(by_scen[0])); w.writeheader(); w.writerows(by_scen)
    print("\nPer run and scenario, seeds leaving the window:")
    for c in learned:
        print(f"  {c:14s}", "  ".join(f"{s}: {sum(1 for r in rows if r['controller']==c and r['scenario']==s and r['peak_excursion']>0)} of "
                                     f"{sum(1 for r in rows if r['controller']==c and r['scenario']==s)}" for s in SCEN))
    # does the combined scenario reproduce the quality scenario, seed by seed?
    q = {(r["controller"], r["seed"]): r for r in rows if r["scenario"] == "quality"}
    c = {(r["controller"], r["seed"]): r for r in rows if r["scenario"] == "combined"}
    d_peak = max(abs(q[k]["peak_excursion"] - c[k]["peak_excursion"]) for k in q if k in c)
    d_cc = max(abs(q[k]["cc_drop"] - c[k]["cc_drop"]) for k in q if k in c)
    print(f"\ncombined vs quality, largest per-seed difference: peak {d_peak:.4f} deg, Cc drop {d_cc:.4f} points")
    print("-> if both are below 0.001, 'reproduces the quality scenario to three decimals' holds for both metrics,")
    print("   and the pooled count is then 2 x the quality count plus any pressure-sag excursions.")


if __name__ == "__main__":
    main()
