#!/usr/bin/env python3
"""
reselect_gate_sweep.py
======================
Does the Pareto front width depend on how strict the thermal safety gate is
at checkpoint selection?  Re-selects checkpoints from the SAVED runs in
runs_multiseed/ under several gate rules and re-evaluates the new pick on
the held-out TEST seeds.  No retraining.

Gates (applied to each checkpoint's validation.csv row):
    paper_gate   all_seeds AND validation ripple (T_std) <= 0.15 C   (the gate stated in Section 4.4)
    all_seeds    steady-state T12 inside [70,78] on ALL validation seeds   (train_multiseed default)
    80pct_seeds  inside the window on >= 80 % of validation seeds
    mean_only    only the mean T12 over validation seeds must be inside     (~ the old Table-3 method)
    no_gate      best validation score, safety ignored

Outputs (in OUT_DIR):
    gate_sweep_results.csv   one row per (gate, algo, pref, seed): chosen ckpt + test metrics
    gate_sweep_summary.csv   per (gate, pref): mean +- sd over seeds
    gate_sweep_width.csv     per gate: front width (max-min of pref means), Kruskal p, worst T_max
Run:
    python reselect_gate_sweep.py --algos SAC --workers 5
    python reselect_gate_sweep.py --mock     (needs `python train_multiseed.py --mock` first)
"""
import os, csv, json, argparse, itertools
import numpy as np
import train_multiseed as tm

RIPPLE_MAX = 0.15   # paper Section 4.4: ripple above 0.15 C rejects a candidate
GATES = {
    "paper_gate":  lambda r: r["safe_frac"] >= 1.0 - 1e-9 and r["T_std"] <= RIPPLE_MAX,
    "all_seeds":   lambda r: r["safe_frac"] >= 1.0 - 1e-9,
    "80pct_seeds": lambda r: r["safe_frac"] >= 0.8 - 1e-9,
    "mean_only":   lambda r: tm.T_LO <= r["T12"] <= tm.T_HI,
    "no_gate":     lambda r: True,
}
NUM = ["ckpt","score","score_std","T12","T_std","T_min","T_max","Cc","Gp","extraction","safe_frac"]


def read_validation(rd):
    rows = []
    for r in csv.DictReader(open(os.path.join(rd, "validation.csv"))):
        d = {k: (float(r[k]) if k in NUM else r[k]) for k in r}
        d["ckpt"] = int(d["ckpt"]); rows.append(d)
    return rows


def process_run(job):
    algo, pi, seed, out_dir = job
    w = tm.PREFERENCE_SET[pi]
    rd = os.path.join(out_dir, algo, f"pref{pi}", f"seed{seed}")
    if not os.path.exists(os.path.join(rd, "validation.csv")):
        return []
    val = read_validation(rd)
    # cache: ckpt -> test aggregate (seed it with the strict-gate result already on disk)
    cache = {}
    rj = os.path.join(rd, "result.json")
    if os.path.exists(rj):
        r = json.load(open(rj)); cache[int(r["ckpt"])] = {k: r[k] for k in ("test_score","Cc","Gp","T12","T_std","T_max","extraction","safe_frac_test")}
    agent = None
    out = []
    for gname, gate in GATES.items():
        pool = [r for r in val if gate(r)]
        gated = bool(pool)
        if not pool: pool = val
        best = max(pool, key=lambda r: r["score"])
        c = best["ckpt"]
        if c not in cache:
            if agent is None:
                env = tm.make_env(seed, w); obs = tm.reset_env(env, seed, w)
                agent = tm.make_agent(algo, obs.shape[0], env.action_space.shape[0], seed)
            agent.load(os.path.join(rd, f"ckpt_{c:06d}.pt"))
            a, _ = tm.evaluate(agent, w, tm.TEST_SEEDS)
            cache[c] = dict(test_score=a["score"], Cc=a["Cc"], Gp=a["Gp"], T12=a["T12"], T_std=a["T_std"],
                            T_max=a["T_max"], extraction=a["extraction"], safe_frac_test=a["safe_frac"])
        t = cache[c]
        out.append(dict(gate=gname, algo=algo, pref_i=pi, seed=seed, ckpt=c, gate_satisfiable=gated,
                        val_score=best["score"], **t))
        print(f"[{gname:<11}] {algo} pref{pi} seed{seed} -> ckpt {c:6d}  T12={t['T12']:.2f} Gp={t['Gp']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=["SAC"])
    ap.add_argument("--prefs", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    out_dir = tm.OUT_DIR
    if a.mock: tm.install_mock(); out_dir = "runs_mock"
    prefs = a.prefs if a.prefs is not None else list(range(len(tm.PREFERENCE_SET)))
    jobs = [(al, pi, s, out_dir) for al, pi, s in itertools.product(a.algos, prefs, tm.TRAIN_SEEDS)]

    rows = []
    if a.workers > 1 and not a.mock:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(process_run, jobs): rows += r
    else:
        for j in jobs: rows += process_run(j)
    if not rows: print("no runs found"); return
    tm.write_csv(os.path.join(out_dir, "gate_sweep_results.csv"), rows)

    # ---- summary per (gate, pref) ----
    summ, width = [], []
    from scipy import stats
    for g in GATES:
        pref_means = {k: [] for k in ("Gp", "T12", "Cc", "extraction")}
        groups_Gp, worst_T, unsat = [], 0.0, 0
        for al, pi in [(a_, p_) for a_ in a.algos for p_ in prefs]:
            grp = [r for r in rows if r["gate"] == g and r["algo"] == al and r["pref_i"] == pi]
            if not grp: continue
            d = dict(gate=g, algo=al, pref_i=pi, preference=str(tm.PREFERENCE_SET[pi]), n=len(grp),
                     gate_unsatisfiable_runs=sum(1 for r in grp if not r["gate_satisfiable"]))
            for k in ("test_score", "Gp", "T12", "Cc", "extraction", "T_max", "safe_frac_test"):
                v = np.array([r[k] for r in grp]); d[f"{k}_mean"] = round(v.mean(), 4); d[f"{k}_sd"] = round(v.std(ddof=1) if len(v) > 1 else 0, 4)
            summ.append(d)
            for k in pref_means: pref_means[k].append(d[f"{k}_mean"])
            if al != a.algos[0]: continue
            groups_Gp.append([r["Gp"] for r in grp]); worst_T = max(worst_T, max(r["T_max"] for r in grp))
            unsat += d["gate_unsatisfiable_runs"]
        def _kw(groups):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore"); return stats.kruskal(*groups).pvalue
            except Exception: return float("nan")
        kw = _kw(groups_Gp) if len(groups_Gp) > 1 else float("nan")
        kwT = _kw([[r["T12"] for r in rows if r["gate"] == g and r["algo"] == a.algos[0] and r["pref_i"] == pi] for pi in prefs]) if len(groups_Gp) > 1 else float("nan")
        width.append(dict(gate=g,
                          Gp_width=round(max(pref_means["Gp"]) - min(pref_means["Gp"]), 3),
                          T12_width=round(max(pref_means["T12"]) - min(pref_means["T12"]), 2),
                          Cc_width=round(max(pref_means["Cc"]) - min(pref_means["Cc"]), 2),
                          extraction_width=round(max(pref_means["extraction"]) - min(pref_means["extraction"]), 4),
                          kruskal_p_Gp=round(kw, 3), kruskal_p_T12=round(kwT, 3),
                          worst_T_max=round(worst_T, 2),
                          mean_safe_frac_test=round(np.mean([r["safe_frac_test"] for r in rows if r["gate"] == g]), 3),
                          runs_where_gate_unsatisfiable=unsat))
    tm.write_csv(os.path.join(out_dir, "gate_sweep_summary.csv"), summ)
    tm.write_csv(os.path.join(out_dir, "gate_sweep_width.csv"), width)
    print("\n" + "=" * 100)
    print(f"{'gate':<13}{'Gp width':>10}{'T12 width':>11}{'Cc width':>10}{'KW p(Gp)':>10}{'KW p(T12)':>11}{'worst Tmax':>12}{'safe(test)':>12}{'unsat.':>8}")
    for wd in width:
        print(f"{wd['gate']:<13}{wd['Gp_width']:>10.3f}{wd['T12_width']:>11.2f}{wd['Cc_width']:>10.2f}{wd['kruskal_p_Gp']:>10.3f}{wd['kruskal_p_T12']:>11.3f}{wd['worst_T_max']:>12.2f}{wd['mean_safe_frac_test']:>12.3f}{wd['runs_where_gate_unsatisfiable']:>8}")
    print("=" * 100)
    print("Reading: if width grows and safe(test) falls as the gate relaxes, the apparent")
    print("flexibility lives in the thermally marginal zone and disappears under robust safety.")


if __name__ == "__main__":
    main()
