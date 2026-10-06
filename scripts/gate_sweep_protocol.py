#!/usr/bin/env python3
"""
gate_sweep_protocol.py  (replaces reselect_gate_sweep.py for Table 8)
=======================================================================
Editor's point 4: Table 8's "strict rule (used)" row did not match Table 3 for
the same five runs. Cause: the old sweep read the training-time validation.csv
and evaluated with the training-time evaluator, while Table 3 comes from the
final protocol (validation_protocol.csv, unified evaluation, DR off). This
script re-runs the sweep on the final protocol pipeline, so the strict row is
Table 3's energy-priority row by construction, and it checks that.

Rules vary only the thermal condition; the ripple gate (<= 0.15 degrees) is
applied under every rule, as Section 5.7 says.

Run from the project folder (same place as reeval_protocol.py), after
reeval_protocol.py --stage select has been run:

    python gate_sweep_protocol.py --pref 3 --algos SAC --workers 5
    python gate_sweep_protocol.py --pref 3 --from-results results/gate_sweep/gate_sweep_all_results.csv

Writes gate_sweep_protocol_results.csv and gate_sweep_protocol_table8.csv and
prints Table 8 together with the Table 3 row it must match.
"""
import os, csv, json, argparse
import numpy as np
from multiprocessing import Pool

import train_multiseed as tm
import reeval_protocol as RP          # FrozenPolicy, eval_seeds, RIPPLE_LIMIT, OUT, UE

T_LO, T_HI = 70.0, 78.0
RULES = {
    "strict":  lambda v: v["max_viol"] <= 0.0 and v["T12_std"] <= RP.RIPPLE_LIMIT,   # = Table 3
    "pct80":   lambda v: v["safe_seed_frac"] >= 0.8 - 1e-9 and v["T12_std"] <= RP.RIPPLE_LIMIT,
    "mean":    lambda v: T_LO <= v["T12"] <= T_HI and v["T12_std"] <= RP.RIPPLE_LIMIT,
    "none":    lambda v: v["T12_std"] <= RP.RIPPLE_LIMIT,
}
LABEL = {"strict": "Inside window on all validation seeds (used)",
         "pct80": "Inside on >= 80 % of validation seeds",
         "mean": "Mean over validation seeds inside",
         "none": "No thermal rule"}
NUM = ["ckpt", "score", "Cc", "Gp", "T12", "T12_std", "extraction", "G_total", "in_window",
       "max_viol", "safe_seed_frac"]


def read_val(rd):
    rows = []
    for r in csv.DictReader(open(os.path.join(rd, "validation_protocol.csv"))):
        d = {k: (float(r[k]) if k in NUM else r[k]) for k in r}
        d["ckpt"] = int(d["ckpt"]); d["admissible"] = r["admissible"] in ("True", "true", "1")
        rows.append(d)
    return rows


def process_run(job):
    algo, pi, seed = job
    w = tm.PREFERENCE_SET[pi]
    rd = os.path.join(RP.OUT, algo, f"pref{pi}", f"seed{seed}")
    vp = os.path.join(rd, "validation_protocol.csv")
    if not os.path.exists(vp):
        raise SystemExit(f"{vp} missing: run  reeval_protocol.py --stage select  first")
    val = read_val(rd)
    installed = json.load(open(os.path.join(rd, "selection_protocol.json")))["ckpt"]
    cache, out = {}, []
    for rule, ok in RULES.items():
        pool = [v for v in val if ok(v)]
        satisfiable = bool(pool)
        best = max(pool if pool else val, key=lambda v: v["score"])
        c = best["ckpt"]
        if rule == "strict" and c != installed:
            raise SystemExit(f"{algo} pref{pi} seed{seed}: strict rule picks ckpt {c} but Table 3 installed {installed}")
        if c not in cache:
            agg, rows = RP.eval_seeds(
                lambda: RP.FrozenPolicy(algo, os.path.join(rd, f"ckpt_{c:06d}.pt"), w, seed), w, tm.TEST_SEEDS)
            cache[c] = dict(test_score=agg["score"], Gp=agg["Gp"], T12=agg["T12"], Cc=agg["Cc"],
                            extraction=agg["extraction"], G_total=agg["G_total"],
                            safe_seed_frac_test=agg["safe_seed_frac"],
                            T12_min_test=min(r["T12_min"] for r in rows))
        out.append(dict(rule=rule, algo=algo, pref_i=pi, seed=seed, ckpt=c, rule_satisfiable=satisfiable,
                        val_score=best["score"], **cache[c]))
        print(f"[{rule:<6}] {algo} pref{pi} seed{seed} -> ckpt {c:6d}  Gp={cache[c]['Gp']:.3f} "
              f"T12={cache[c]['T12']:.2f} safe={cache[c]['safe_seed_frac_test']:.2f}", flush=True)
    return out


def build_table8(res):
    """Table 8 from per-run results. The last column is the LEAST SAFE run (lowest test
    safe fraction, ties broken by the lower temperature), as the paper reports it."""
    table = []
    print("\nTable 8, rebuilt on the final protocol (mean +- sd over the five runs, test seeds)")
    print(f"{'rule':46s}{'Gp, kg/s':>16}{'T12, C':>16}{'safe frac':>11}{'least safe run T12/safe':>25}")
    for rule in RULES:
        g = [r for r in res if r["rule"] == rule]
        Gp = np.array([float(r["Gp"]) for r in g]); T = np.array([float(r["T12"]) for r in g])
        sf = np.array([float(r["safe_seed_frac_test"]) for r in g])
        least = min(g, key=lambda r: (float(r["safe_seed_frac_test"]), float(r["T12"])))
        row = dict(rule=LABEL[rule], Gp_mean=round(Gp.mean(), 3), Gp_sd=round(Gp.std(ddof=1), 3),
                   T12_mean=round(T.mean(), 2), T12_sd=round(T.std(ddof=1), 2),
                   safe_frac=round(sf.mean(), 2), least_safe_T12=round(float(least["T12"]), 2),
                   least_safe_frac=round(float(least["safe_seed_frac_test"]), 2),
                   runs_rule_unsatisfiable=sum(1 for r in g if str(r["rule_satisfiable"]) in ("False", "false", "0")))
        table.append(row)
        print(f"{row['rule']:46s}{row['Gp_mean']:>8.3f} +- {row['Gp_sd']:<5.3f}{row['T12_mean']:>8.2f} +- {row['T12_sd']:<5.2f}"
              f"{row['safe_frac']:>11.2f}{row['least_safe_T12']:>15.2f} / {row['least_safe_frac']:<5.2f}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref", type=int, default=3)
    ap.add_argument("--algos", nargs="*", default=["SAC"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(tm.TRAIN_SEEDS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--from-results", default=None,
                    help="rebuild Table 8 from a stored per-run CSV without re-evaluating anything")
    a = ap.parse_args()
    if a.from_results:
        res = [r for r in csv.DictReader(open(a.from_results)) if int(r["pref_i"]) == a.pref and r["algo"] in a.algos]
        tm.write_csv("gate_sweep_protocol_table8.csv", build_table8(res))
        return
    jobs = [(al, a.pref, s) for al in a.algos for s in a.seeds]
    with Pool(a.workers) as p:
        res = [r for rr in p.map(process_run, jobs) for r in rr]
    tm.write_csv("gate_sweep_protocol_results.csv", res)

    table = build_table8(res)
    tm.write_csv("gate_sweep_protocol_table8.csv", table)

    # ---- the row Table 3 reports for the same preference, from protocol_results.csv ----
    pr = os.path.join(RP.OUT, "protocol_results.csv")
    if os.path.exists(pr):
        rows = [r for r in csv.DictReader(open(pr)) if r["algo"] in a.algos and int(r["pref_i"]) == a.pref]
        Gp = np.array([float(r["Gp"]) for r in rows]); T = np.array([float(r["T12"]) for r in rows])
        sf = np.array([float(r["safe_seed_frac_test"]) for r in rows])
        print(f"\nTable 3 row for this preference: Gp {Gp.mean():.3f} +- {Gp.std(ddof=1):.3f} | "
              f"T12 {T.mean():.2f} +- {T.std(ddof=1):.2f} | safe {sf.mean():.2f}")
        strict = table[0]
        same = abs(strict["Gp_mean"] - Gp.mean()) < 5e-4 and abs(strict["T12_mean"] - T.mean()) < 5e-3
        print("strict row == Table 3 row:", "YES" if same else "NO  <- investigate before editing the paper")


if __name__ == "__main__":
    main()
