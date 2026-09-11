#!/usr/bin/env python3
"""
reeval_protocol.py
==================
Re-evaluates every saved checkpoint in runs_multiseed/ under the paper's
unified evaluation protocol, so that the learned controllers (Tables 3, 7)
and the classical cascade (Tables 4, 5) are measured on the same plant with
the same gate. No retraining.

Protocol (unified_evaluation.py):
    env       : ENV_KW = enable_disturbance=True, domain_randomize=False
    metrics   : run_episode windows AVG_WINDOW / STD_WINDOW / SETTLE_SKIP
    gate      : admissible iff max_viol == 0 on EVERY validation seed
                (no excursion outside [T_LO, T_HI] after SETTLE_SKIP)
                AND mean validation ripple T12_std <= RIPPLE_LIMIT
    selection : highest u_w score among admissible checkpoints (validation seeds)
    reporting : frozen pick evaluated ONCE on the test seeds

Seed sets (train_multiseed.py): validation 100-109, test 200-219.

Stage 1  --stage select   re-select + re-test every run           -> runs_multiseed/protocol_results.csv
                                                                     runs_multiseed/protocol_summary.csv
Stage 2  --stage classical classical grid on the TEST seeds        -> runs_multiseed/classical_grid_test.csv
Stage 3  --stage matched  matched-energy paired comparison         -> runs_multiseed/matched_comparison.csv
                                                                     runs_multiseed/matched_summary.csv
Stage 4  --stage headline the rows of Table 5 exactly              -> runs_multiseed/headline_table5.csv
Stage 5  --stage all      1 -> 2 -> 3 -> 4

Run in the project folder:
    python reeval_protocol.py --stage all --workers 5
"""
import os, sys, csv, json, argparse, itertools
import numpy as np

import train_multiseed as tm
import unified_evaluation as UE
import morl_score as MS
from sugar_extraction_env import SugarExtractionEnv

RIPPLE_LIMIT = 0.15
OUT = tm.OUT_DIR
ALGOS_ALL = ["SAC", "TD3", "PPO"]

# ---------------------------------------------------------------- learned policy shim
class FrozenPolicy:
    """Wraps a train_multiseed checkpoint with the controller interface unified_evaluation expects."""
    def __init__(self, algo, ckpt_path, preference, seed=0):
        env = SugarExtractionEnv(seed=seed, **UE.ENV_KW)
        obs, _ = env.reset(seed=seed, options={"preference": preference})
        self.agent = tm.make_agent(algo, obs.shape[0], env.action_space.shape[0], seed, preference)
        self.agent.load(ckpt_path)
    def reset(self): pass
    def act(self, obs, info): return self.agent.act(obs, deterministic=True)


def episode_with_return(env, ctrl, seed, w):
    """unified_evaluation.run_episode metrics + the scalarized return u_w (needed for selection)."""
    obs, _ = env.reset(seed=seed, options={"preference": w})
    ctrl.reset()
    info = dict(env.model.get_outputs(env.x), Kc=env.model.Kc)
    T12, Cc, Gp, q = [], [], [], None; done = False
    while not done:
        a = ctrl.act(obs, info)
        obs, _, term, trunc, info = env.step(a)
        rv = np.asarray(info["reward_vec"], float); q = rv if q is None else q + rv
        T12.append(info["T12"]); Cc.append(info["Cc"]); Gp.append(info["Gp"]); done = term or trunc
    T12 = np.asarray(T12); Cc = np.asarray(Cc); Gp = np.asarray(Gp)
    Tsafe = T12[UE.SETTLE_SKIP:] if len(T12) > UE.SETTLE_SKIP else T12
    viol = np.maximum(np.maximum(UE.T_LO - Tsafe, 0.0), np.maximum(Tsafe - UE.T_HI, 0.0))
    m = dict(score=float(tm.u_w(q, w)),
             Cc=float(np.mean(Cc[-UE.AVG_WINDOW:])), Gp=float(np.mean(Gp[-UE.AVG_WINDOW:])),
             T12=float(np.mean(T12[-UE.AVG_WINDOW:])), T12_std=float(np.std(T12[-UE.STD_WINDOW:])),
             extraction=float(info["extraction"]) * 100.0,
             max_viol=float(np.max(viol)), in_window=float(np.mean(viol == 0.0)) * 100.0,
             T12_min=float(np.min(Tsafe)), T12_max=float(np.max(Tsafe)))
    m["G_total"] = float(MS.total_steam(m["Gp"], m["extraction"], m["Cc"]))
    return m


def eval_seeds(ctrl_factory, w, seeds):
    rows = [episode_with_return(SugarExtractionEnv(seed=s, **UE.ENV_KW), ctrl_factory(), s, w) for s in seeds]
    keys = ["score", "Cc", "Gp", "T12", "T12_std", "extraction", "G_total", "in_window"]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    agg["max_viol"] = float(max(r["max_viol"] for r in rows))
    agg["admissible"] = bool(agg["max_viol"] <= 0.0 and agg["T12_std"] <= RIPPLE_LIMIT)
    agg["safe_seed_frac"] = float(np.mean([r["max_viol"] <= 0.0 for r in rows]))
    return agg, rows


# ---------------------------------------------------------------- stage 1: reselect + retest
def reselect_run(job):
    algo, pi, seed = job
    w = tm.PREFERENCE_SET[pi]
    rd = os.path.join(OUT, algo, f"pref{pi}", f"seed{seed}")
    if not os.path.isdir(rd): return None
    ckpts = sorted(int(f[5:11]) for f in os.listdir(rd) if f.startswith("ckpt_"))
    if not ckpts: return None
    val = []
    for c in ckpts:
        agg, _ = eval_seeds(lambda: FrozenPolicy(algo, os.path.join(rd, f"ckpt_{c:06d}.pt"), w, seed), w, tm.VAL_SEEDS)
        val.append(dict(ckpt=c, **agg))
    tm.write_csv(os.path.join(rd, "validation_protocol.csv"), val)
    pool = [v for v in val if v["admissible"]]
    any_adm = bool(pool)
    best = max(pool if pool else val, key=lambda v: v["score"])
    tagg, trows = eval_seeds(lambda: FrozenPolicy(algo, os.path.join(rd, f"ckpt_{best['ckpt']:06d}.pt"), w, seed), w, tm.TEST_SEEDS)
    tm.write_csv(os.path.join(rd, "test_protocol.csv"), [dict(seed=s, **r) for s, r in zip(tm.TEST_SEEDS, trows)])
    sel = dict(algo=algo, pref_i=pi, preference=w, seed=seed, ckpt=best["ckpt"], any_admissible=any_adm,
               protocol="unified_evaluation: DR=False, max_viol==0 all val seeds, ripple<=%.2f" % RIPPLE_LIMIT,
               val_seeds=tm.VAL_SEEDS, test_seeds=tm.TEST_SEEDS)
    json.dump(sel, open(os.path.join(rd, "selection_protocol.json"), "w"), indent=2)
    res = dict(algo=algo, pref_i=pi, seed=seed, ckpt=best["ckpt"], any_admissible=any_adm,
               val_score=best["score"], test_score=tagg["score"], offset=tagg["score"] - best["score"],
               Cc=tagg["Cc"], Gp=tagg["Gp"], G_total=tagg["G_total"], T12=tagg["T12"], ripple=tagg["T12_std"],
               extraction=tagg["extraction"], max_viol_test=tagg["max_viol"], safe_seed_frac_test=tagg["safe_seed_frac"])
    print(f"[{algo} pref{pi} seed{seed}] ckpt {best['ckpt']:6d} adm={any_adm}  Gtot={res['G_total']:.3f}  "
          f"eta={res['extraction']:.2f}  T={res['T12']:.2f}  viol={res['max_viol_test']:.2f}", flush=True)
    return res


def stage_select(algos, workers):
    jobs = [(a, p, s) for a, p, s in itertools.product(algos, range(len(tm.PREFERENCE_SET)), tm.TRAIN_SEEDS)
            if os.path.isdir(os.path.join(OUT, a, f"pref{p}", f"seed{s}"))]
    print(f"stage 1: {len(jobs)} runs, {len(tm.VAL_SEEDS)} val + {len(tm.TEST_SEEDS)} test seeds, DR=False")
    rows = []
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(reselect_run, jobs):
                if r: rows.append(r)
    else:
        rows = [r for r in map(reselect_run, jobs) if r]
    tm.write_csv(os.path.join(OUT, "protocol_results.csv"), rows)
    summ = []
    for a in algos:
        for p in range(len(tm.PREFERENCE_SET)):
            g = [r for r in rows if r["algo"] == a and r["pref_i"] == p]
            if not g: continue
            d = dict(algo=a, pref_i=p, preference=str(tm.PREFERENCE_SET[p]), n=len(g),
                     runs_without_admissible=sum(1 for r in g if not r["any_admissible"]))
            for k in ["test_score", "G_total", "Gp", "Cc", "T12", "extraction", "ripple", "offset", "safe_seed_frac_test"]:
                v = np.array([r[k] for r in g]); d[f"{k}_mean"] = round(v.mean(), 4); d[f"{k}_sd"] = round(v.std(ddof=1) if len(v) > 1 else 0, 4)
            d["max_viol_test_worst"] = round(max(r["max_viol_test"] for r in g), 3)
            summ.append(d)
    tm.write_csv(os.path.join(OUT, "protocol_summary.csv"), summ)
    print("\nprotocol_summary.csv")
    print(f"{'algo':<5}{'pref':<5}{'n':<3}{'score':>18}{'G_total':>15}{'eta':>13}{'T12':>13}{'ripple':>8}{'viol':>6}")
    for d in summ:
        print(f"{d['algo']:<5}{d['pref_i']:<5}{d['n']:<3}{d['test_score_mean']:>10.1f}±{d['test_score_sd']:<7.1f}"
              f"{d['G_total_mean']:>8.3f}±{d['G_total_sd']:<6.3f}{d['extraction_mean']:>7.2f}±{d['extraction_sd']:<5.2f}"
              f"{d['T12_mean']:>7.2f}±{d['T12_sd']:<5.2f}{d['ripple_mean']:>8.3f}{d['max_viol_test_worst']:>6.2f}")
    return rows


# ---------------------------------------------------------------- stage 2: classical grid on TEST seeds
def stage_classical(workers):
    from baseline_controllers_mv import PIDControllerMV, MPCControllerMV, FuzzyControllerMV
    import classical_grid_sweep as CG
    ctrls = [("PID", PIDControllerMV), ("MPC", MPCControllerMV), ("Fuzzy", FuzzyControllerMV)]
    w = [0.34, 0.33, 0.33]
    rows = []
    for name, cls in ctrls:
        for T in CG.T_TARGETS:
            for cc in CG.CC_TARGETS:
                agg, per = eval_seeds(lambda: CG.build(cls, SugarExtractionEnv(seed=0, **UE.ENV_KW), T, cc), w, tm.TEST_SEEDS)
                rows.append(dict(controller=name, T_set=T, Cc_set=cc, **{k: agg[k] for k in ("Cc", "Gp", "G_total", "T12", "extraction", "T12_std", "max_viol")},
                                 admissible=int(agg["admissible"]),
                                 per_seed_extraction=json.dumps([round(r["extraction"], 4) for r in per]),
                                 per_seed_G_total=json.dumps([round(r["G_total"], 4) for r in per])))
        print(f"  classical {name} done", flush=True)
    tm.write_csv(os.path.join(OUT, "classical_grid_test.csv"), rows)
    adm = [r for r in rows if r["admissible"]]
    print(f"stage 2: admissible classical points {len(adm)} of {len(rows)}; "
          f"G_total {min(r['G_total'] for r in adm):.3f}..{max(r['G_total'] for r in adm):.3f}; "
          f"extraction {min(r['extraction'] for r in adm):.3f}..{max(r['extraction'] for r in adm):.3f}")
    return rows


# ---------------------------------------------------------------- stage 3: matched-energy paired comparison
def stage_matched(match_tol=0.03):
    from scipy.stats import wilcoxon
    P = list(csv.DictReader(open(os.path.join(OUT, "protocol_results.csv"))))
    C = list(csv.DictReader(open(os.path.join(OUT, "classical_grid_test.csv"))))
    C = [dict(r, G_total=float(r["G_total"]), extraction=float(r["extraction"]),
              pe=json.loads(r["per_seed_extraction"]), pg=json.loads(r["per_seed_G_total"]))
         for r in C if r["admissible"] == "1"]
    out = []
    for r in P:
        if r["algo"] != "SAC": continue
        pi, seed = int(r["pref_i"]), int(r["seed"])
        rd = os.path.join(OUT, "SAC", f"pref{pi}", f"seed{seed}")
        L = list(csv.DictReader(open(os.path.join(rd, "test_protocol.csv"))))
        Le = np.array([float(x["extraction"]) for x in L]); Lg = np.array([float(x["G_total"]) for x in L])
        for name in ("PID", "MPC", "Fuzzy"):
            cand = [c for c in C if c["controller"] == name]
            best = min(cand, key=lambda c: abs(c["G_total"] - Lg.mean()))
            dg = best["G_total"] - Lg.mean(); de = Le - np.array(best["pe"])
            p = wilcoxon(Le, np.array(best["pe"]), alternative="two-sided").pvalue if not np.all(de == 0) else 1.0
            out.append(dict(pref_i=pi, preference=str(tm.PREFERENCE_SET[pi]), seed=seed, baseline=name,
                            T_set=best["T_set"], Cc_set=best["Cc_set"], learned_G_total=round(Lg.mean(), 4),
                            classical_G_total=round(best["G_total"], 4), d_G_total=round(dg, 4), matched=int(abs(dg) <= match_tol),
                            d_extraction=round(float(de.mean()), 4), p_wilcoxon_20seeds=round(float(p), 4),
                            learned_wins_pct=round(100 * float(np.mean(de > 0)), 0)))
    tm.write_csv(os.path.join(OUT, "matched_comparison.csv"), out)
    # aggregate across the five training seeds: the primary statistic (training variability)
    summ = []
    from math import isnan
    for pi in sorted({o["pref_i"] for o in out}):
        for name in ("PID", "MPC", "Fuzzy"):
            g = [o for o in out if o["pref_i"] == pi and o["baseline"] == name]
            de = np.array([o["d_extraction"] for o in g]); dg = np.array([o["d_G_total"] for o in g])
            # sign test across training seeds + Wilcoxon on the 5 seed-level differences
            p5 = wilcoxon(de, alternative="two-sided").pvalue if len(de) > 1 and not np.all(de == 0) else float("nan")
            summ.append(dict(pref_i=pi, preference=str(tm.PREFERENCE_SET[pi]), baseline=name, n_seeds=len(g),
                             d_extraction_mean=round(de.mean(), 4), d_extraction_sd=round(de.std(ddof=1) if len(de) > 1 else 0, 4),
                             d_G_total_mean=round(dg.mean(), 4), all_matched=int(all(o["matched"] for o in g)),
                             seeds_learned_ahead=int(np.sum(de > 0)), p_wilcoxon_5seeds=round(float(p5), 4) if not isnan(p5) else "nan"))
    tm.write_csv(os.path.join(OUT, "matched_summary.csv"), summ)
    print("\nmatched_summary.csv  (learned - classical extraction, pp, at matched plant-level energy)")
    print(f"{'pref':<6}{'baseline':<9}{'Δη mean±sd':>16}{'ΔG':>9}{'matched':>9}{'seeds ahead':>13}{'p(5 seeds)':>12}")
    for d in summ:
        print(f"{d['pref_i']:<6}{d['baseline']:<9}{d['d_extraction_mean']:>+9.3f}±{d['d_extraction_sd']:<6.3f}"
              f"{d['d_G_total_mean']:>+9.3f}{d['all_matched']:>9}{d['seeds_learned_ahead']:>8}/{d['n_seeds']:<4}{str(d['p_wilcoxon_5seeds']):>12}")
    return summ



# ---------------------------------------------------------------- stage 4: the headline rows of Table 5
COLD_PREFS = [2, 3, 4, 5]          # the preferences that settle on the cold arc (Section 5.2)
HOT_PREF   = 0                     # extraction priority
MATCH_TOL  = 0.03                  # kg/s of plant-level steam
MIN_PAIRS_FOR_TEST = 5             # below this, 2/2**n > 0.05 and no test can be significant

def stage_headline():
    """Reproduces Table 5 exactly.

    Two differences from stage_matched, and both matter:
      1. only ENERGY-MATCHED pairs enter, |dG| <= MATCH_TOL. A pair at two different
         energies measures the energy, not the controller.
      2. the signed-rank test is taken over the matched PAIRS, not over the five
         training runs. With five runs the smallest attainable two-sided p is
         2/2^5 = 0.0625, so no five-run test could ever reach significance; the
         pairs are the unit that carries the evidence.
    """
    from scipy.stats import wilcoxon
    rows = list(csv.DictReader(open(os.path.join(OUT, "matched_comparison.csv"))))
    for r in rows:
        r["pref_i"] = int(r["pref_i"]); r["matched"] = int(r["matched"])
        r["d_extraction"] = float(r["d_extraction"]); r["d_G_total"] = float(r["d_G_total"])
        r["p_wilcoxon_20seeds"] = float(r["p_wilcoxon_20seeds"])

    def block(label, sel):
        m = [r for r in sel if r["matched"] == 1]
        if not m:
            return dict(group=label, matched_pairs=f"0 of {len(sel)}", d_G_total="n/a",
                        d_extraction="n/a", learned_ahead="n/a", p_pairs="n/a", p_within_run="n/a")
        de = np.array([r["d_extraction"] for r in m])
        dg = np.array([r["d_G_total"] for r in m])
        # With n pairs the smallest attainable two-sided p is 2/2**n, so a test on
        # fewer than five pairs cannot reach 0.05 and is reported as not applicable.
        p = (wilcoxon(de, alternative="two-sided").pvalue
             if len(de) >= MIN_PAIRS_FOR_TEST and not np.all(de == 0) else float("nan"))
        sig = sum(1 for r in m if r["p_wilcoxon_20seeds"] < 0.01)
        return dict(group=label,
                    matched_pairs=f"{len(m)} of {len(sel)}",
                    d_G_total=f"{dg.mean():+.3f}",
                    d_extraction=(f"{de.mean():+.3f} \u00b1 {de.std(ddof=1):.3f}" if len(de) > 1
                                  else f"{de.mean():+.3f}"),
                    learned_ahead=f"{int((de > 0).sum())} of {len(de)}",
                    p_pairs=("not applicable" if np.isnan(p) else f"{p:.3f}"),
                    p_within_run=f"< 0.01 in {sig} of {len(m)}")

    table = []
    for name in ("MPC", "Fuzzy", "PID"):
        table.append(block(f"{name}, cold cluster",
                           [r for r in rows if r["baseline"] == name and r["pref_i"] in COLD_PREFS]))
        if name == "MPC":
            table.append(block("MPC, balanced only",
                               [r for r in rows if r["baseline"] == name and r["pref_i"] == 2]))
    table.append(block("MPC, extraction priority",
                       [r for r in rows if r["baseline"] == "MPC" and r["pref_i"] == HOT_PREF]))
    tm.write_csv(os.path.join(OUT, "headline_table5.csv"), table)

    print("\nTable 5, reproduced (learned minus classical extraction, percentage points,")
    print("at matched plant-level energy; matched means |dG| <= %.2f kg/s)" % MATCH_TOL)
    print("-" * 114)
    print(f"{'group':26s}{'pairs':>12}{'d energy':>11}{'d extraction':>20}"
          f"{'learned ahead':>16}{'p, pairs':>17}{'p, within run':>20}")
    print("-" * 114)
    for d in table:
        print(f"{d['group']:26s}{d['matched_pairs']:>12}{d['d_G_total']:>11}"
              f"{d['d_extraction']:>20}{d['learned_ahead']:>16}{d['p_pairs']:>17}{d['p_within_run']:>20}")
    print("-" * 114)
    print("These rows are Table 5 of the paper. The paired within-run tests use the twenty")
    print("test seeds; the pair-level test uses the matched pairs listed under 'pairs'.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "classical", "matched", "headline", "all"], default="all")
    ap.add_argument("--algos", nargs="*", default=ALGOS_ALL)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.stage in ("select", "all"):   stage_select(a.algos, a.workers)
    if a.stage in ("classical", "all"): stage_classical(a.workers)
    if a.stage in ("matched", "all"):  stage_matched()
    if a.stage in ("headline", "all"): stage_headline()
