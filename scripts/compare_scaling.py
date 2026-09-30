#!/usr/bin/env python3
"""
compare_scaling.py
==================
Compares the delivered policies trained under ORACLE scaling with those
trained under TRAINING-DATA scaling (R2-6), under the same protocol.

Everything is compared in physical units: steam, total energy, temperature,
extraction and concentration. The scalarized score is NOT compared, because
it is defined by the scaling itself and so differs between the two runs by
construction.

Inputs (each written by reeval_protocol.py --stage select):
    runs_multiseed/protocol_results.csv            oracle scaling
    runs_multiseed_training/protocol_results.csv   training-data scaling
    runs_multiseed/classical_grid_test.csv         classical grid (physical,
                                                   identical for both)

The reading below was fixed before the experiment was run, so that the
outcome cannot be reinterpreted after the fact:

  A  preferences still distinct, front of similar width, and the gap to the
     classical front keeps its sign
       -> the conclusions do not depend on oracle scaling
  B  preferences no longer distinct, or the front collapses
       -> oracle scaling was doing real work; scaling is a precondition for
          separable preferences, as Section 4.3 claims, and deployment needs
          a scaling procedure of its own
  In both cases the comparison with the classical cascade is reported, since
  it is physical and does not depend on the scaling.

Run:  python compare_scaling.py
Writes scaling_comparison.csv and prints the verdict.
"""
import os, csv, sys
import numpy as np
from scipy import stats

ORA = os.environ.get("ORACLE_DIR", "runs_multiseed")
TRN = os.environ.get("TRAINING_DIR", "runs_multiseed_training")
PREF = {0: [0.70, 0.10, 0.20], 1: [0.50, 0.30, 0.20], 2: [0.34, 0.33, 0.33],
        3: [0.20, 0.60, 0.20], 4: [0.20, 0.20, 0.60], 5: [0.29, 0.51, 0.20]}
KEYS = ("G_total", "Gp", "T12", "extraction", "Cc")


def load(folder):
    f = os.path.join(folder, "protocol_results.csv")
    if not os.path.exists(f):
        sys.exit(f"missing {f}: run reeval_protocol.py --stage select with MORL_OUT_DIR={folder}")
    rows = [r for r in csv.DictReader(open(f)) if r["algo"] == "SAC"]
    for r in rows:
        for k in KEYS + ("safe_seed_frac_test", "max_viol_test"):
            r[k] = float(r[k])
        r["pref_i"] = int(r["pref_i"])
        r["any_admissible"] = r["any_admissible"] in ("True", "true", "1")
    return rows


def staircase(pts):
    xs, ys = [], []
    for x, y in sorted(pts):
        if not ys or y > ys[-1]:
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def gap_to_classical(rows, grid):
    by = {}
    for r in rows: by.setdefault(r["pref_i"], []).append((r["G_total"], r["extraction"]))
    pts = [(np.mean([g for g, _ in v]), np.mean([e for _, e in v])) for v in by.values()]
    lx, ly = staircase(pts); cx, cy = staircase(grid)
    lo, hi = max(lx.min(), cx.min()), min(lx.max(), cx.max())
    if hi <= lo: return float("nan"), float("nan"), 0.0
    g = np.linspace(lo, hi, 60)
    d = np.interp(g, lx, ly) - np.interp(g, cx, cy)
    return float(d.mean()), float(hi - lo), float(np.mean(d > 0))


def summarise(rows, label):
    prefs = sorted({r["pref_i"] for r in rows})
    out = {"label": label, "prefs": prefs}
    for k in ("G_total", "T12", "extraction"):
        groups = [[r[k] for r in rows if r["pref_i"] == p] for p in prefs]
        out[f"kw_{k}"] = stats.kruskal(*groups).pvalue if len(groups) > 1 else float("nan")
        means = [np.mean(g) for g in groups]
        out[f"width_{k}"] = float(max(means) - min(means))
    wext = [PREF[p][0] for p in prefs]
    t12 = [np.mean([r["T12"] for r in rows if r["pref_i"] == p]) for p in prefs]
    out["spearman_wext_T12"] = stats.spearmanr(wext, t12)[0] if len(prefs) > 2 else float("nan")
    out["admissible_runs"] = f"{sum(r['any_admissible'] for r in rows)}/{len(rows)}"
    out["safe_frac"] = float(np.mean([r["safe_seed_frac_test"] for r in rows]))
    out["worst_viol"] = float(max(r["max_viol_test"] for r in rows))
    return out


def main():
    ora, trn = load(ORA), load(TRN)
    common = sorted({r["pref_i"] for r in ora} & {r["pref_i"] for r in trn})
    ora = [r for r in ora if r["pref_i"] in common]
    trn = [r for r in trn if r["pref_i"] in common]
    gfile = os.path.join(ORA, "classical_grid_test.csv")
    grid = [(float(r["G_total"]), float(r["extraction"])) for r in csv.DictReader(open(gfile))
            if r["admissible"] == "1"]

    so, st = summarise(ora, "oracle"), summarise(trn, "training")
    go, gt = gap_to_classical(ora, grid), gap_to_classical(trn, grid)

    print("=" * 96)
    print(f"Preferences compared: {[PREF[p] for p in common]}")
    print("=" * 96)
    print(f"{'pref':22s}{'scaling':10s}{'G_total':>16}{'T12':>15}{'extraction %':>17}{'Cc %':>14}")
    rows_csv = []
    for p in common:
        for lab, rs in (("oracle", ora), ("training", trn)):
            g = [r for r in rs if r["pref_i"] == p]
            m = {k: (np.mean([r[k] for r in g]), np.std([r[k] for r in g], ddof=1)) for k in KEYS}
            print(f"{str(PREF[p]):22s}{lab:10s}"
                  f"{m['G_total'][0]:>9.3f}\u00b1{m['G_total'][1]:<6.3f}{m['T12'][0]:>8.2f}\u00b1{m['T12'][1]:<5.2f}"
                  f"{m['extraction'][0]:>10.2f}\u00b1{m['extraction'][1]:<5.2f}{m['Cc'][0]:>8.2f}\u00b1{m['Cc'][1]:<4.2f}")
            rows_csv.append(dict(pref=str(PREF[p]), scaling=lab, n=len(g),
                                 **{f"{k}_mean": round(m[k][0], 4) for k in KEYS},
                                 **{f"{k}_sd": round(m[k][1], 4) for k in KEYS}))
    print("-" * 96)
    print(f"{'':32s}{'oracle':>16}{'training':>16}")
    for k in ("G_total", "T12", "extraction"):
        print(f"{'front width, ' + k:32s}{so['width_' + k]:>16.3f}{st['width_' + k]:>16.3f}")
    for k in ("G_total", "T12", "extraction"):
        print(f"{'Kruskal-Wallis p, ' + k:32s}{so['kw_' + k]:>16.4f}{st['kw_' + k]:>16.4f}")
    print(f"{'Spearman w_ext vs T12':32s}{so['spearman_wext_T12']:>16.2f}{st['spearman_wext_T12']:>16.2f}")
    print(f"{'runs with admissible checkpoint':32s}{so['admissible_runs']:>16}{st['admissible_runs']:>16}")
    print(f"{'safe fraction of test seeds':32s}{so['safe_frac']:>16.3f}{st['safe_frac']:>16.3f}")
    print(f"{'learned minus classical, points':32s}{go[0]:>+16.3f}{gt[0]:>+16.3f}")
    print(f"{'  over shared energy range, kg/s':32s}{go[1]:>16.3f}{gt[1]:>16.3f}")
    print(f"{'  fraction learned ahead':32s}{go[2]:>16.2f}{gt[2]:>16.2f}")

    # verdict, by the rule stated in the docstring
    distinct = all(st[f"kw_{k}"] < 0.05 for k in ("G_total", "T12"))
    wide = st["width_G_total"] >= 0.5 * so["width_G_total"]
    same_sign = (np.sign(go[0]) == np.sign(gt[0])) if not np.isnan(gt[0]) else False
    print("=" * 96)
    if distinct and wide and same_sign:
        v = ("A: under training-data scaling the preferences remain distinct, the front keeps at least half "
             "its width, and the gap to the classical cascade keeps its sign. The conclusions do not depend "
             "on oracle scaling.")
    else:
        why = [s for s, c in (("preferences not distinct", not distinct),
                              ("front narrower than half", not wide),
                              ("gap to the cascade changed sign", not same_sign)) if c]
        v = ("B: " + "; ".join(why) + ". Oracle scaling was doing real work: scaling is a precondition for "
             "separable preferences, and a deployable controller needs its own scaling procedure. The "
             "comparison with the cascade is physical and is reported either way.")
    print("VERDICT", v)
    print("=" * 96)
    with open("scaling_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    with open("scaling_verdict.txt", "w") as f:
        f.write(v + "\n")
        for lab, s, g in (("oracle", so, go), ("training", st, gt)):
            f.write(f"{lab}: widths G {s['width_G_total']:.3f} T {s['width_T12']:.3f} | "
                    f"KW G {s['kw_G_total']:.4f} T {s['kw_T12']:.4f} | gap {g[0]:+.3f} | "
                    f"admissible {s['admissible_runs']} | safe {s['safe_frac']:.3f}\n")
    print("wrote scaling_comparison.csv and scaling_verdict.txt")


if __name__ == "__main__":
    main()
