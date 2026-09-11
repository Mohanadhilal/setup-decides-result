#!/usr/bin/env python3
"""
robustness_protocol.py
======================
Re-runs the disturbance study of Section 5.5 under the protocol of Section 4.4,
replacing the three-seed averages with the twenty held-out test seeds and adding
the worst-case and tail statistics the reviewer asked for (R2-7).

What it does
------------
For every scenario, every controller and each of the twenty test seeds:
  * runs one episode with a step disturbance injected one third of the way in,
  * records the transient response after the shock,
and reports, per controller and scenario:

  peak          worst thermal excursion beyond the window, over the whole run
  peak_P95      95th percentile of the per-seed peak (tail, not the mean)
  peak_worst    the worst single seed  (the number a safety case must use)
  t_out         percentage of post-shock time spent outside the window
  seeds_unsafe  how many of the twenty seeds left the window at all
  cc_drop       worst juice-concentration disruption after the shock
  recovery      steps to return inside the window (nan if it never left)

All controllers see the SAME seeds, so the learned-vs-classical differences are
paired and a Wilcoxon signed-rank test over the twenty seeds is reported.

Scenarios (as in the submitted Section 5.5):
  quality   raw-material quality Kc 1.0 -> 0.75
  pressure  steam supply pressure -20 %
  combined  both at once

Outputs (under runs_multiseed/):
  robustness_perseed.csv    one row per scenario x controller x seed
  robustness_summary.csv    aggregated, including the tail statistics
  robustness_tests.csv      paired learned-vs-baseline tests per scenario

Run in the project folder, after reeval_protocol.py --stage select:
    python robustness_protocol.py --pref 2 --workers 5
"""
import os, csv, json, argparse, itertools
import numpy as np

import train_multiseed as tm
import unified_evaluation as UE
import morl_score as MS
from sugar_extraction_env import SugarExtractionEnv
from reeval_protocol import FrozenPolicy

SCENARIOS = ["quality", "pressure", "combined"]
T_TARGET  = 73.5
OUT       = tm.OUT_DIR

# classical baselines with the draft loop, at the set points used in Section 5.3
from baseline_controllers_mv import PIDControllerMV, MPCControllerMV, FuzzyControllerMV
CLASSICAL = [("PID", PIDControllerMV), ("MPC", MPCControllerMV), ("Fuzzy", FuzzyControllerMV)]
T_SET, CC_SET = 73.0, 12.10          # the admissible grid point nearest the cold cluster


def build_classical(cls, env):
    if cls is MPCControllerMV:
        return cls(env.model, T_target=T_SET, Cc_target=CC_SET, dt=env.dt)
    return cls(T_target=T_SET, Cc_target=CC_SET, dt=env.dt)


def run_shock(env, ctrl, scenario, seed, w):
    """One episode with a step disturbance at one third of the horizon."""
    obs, _ = env.reset(seed=seed, options={"preference": w})
    ctrl.reset()
    info = dict(env.model.get_outputs(env.x), Kc=env.model.Kc,
                Pp=env.model.P_p, Gv=env.model.G_v)
    shock_at = env.episode_steps // 3
    T, Cc, step, done = [], [], 0, False
    while not done:
        if step == shock_at:
            if scenario in ("quality", "combined"):
                env.model.Kc_nom = 0.75; env.model.Kc = 0.75
            if scenario in ("pressure", "combined"):
                env.model.P_p_nom = 3.5e5 * 0.80; env.model.P_p = 3.5e5 * 0.80
        a = ctrl.act(obs, info)
        obs, _, term, trunc, info = env.step(a)
        T.append(float(info["T12"])); Cc.append(float(info["Cc"]))
        done = term or trunc; step += 1
    env.model.Kc_nom = 1.0; env.model.P_p_nom = 3.5e5      # restore for the next run

    T = np.asarray(T); Cc = np.asarray(Cc)
    post = slice(shock_at, len(T))
    Tp = T[post]
    out = np.maximum(np.maximum(UE.T_LO - Tp, 0.0), np.maximum(Tp - UE.T_HI, 0.0))
    left = out > 0
    if left.any():                                    # steps until it is back inside for good
        last = int(np.max(np.nonzero(left)[0]))
        recovery = float(last + 1)
    else:
        recovery = float("nan")
    return dict(
        peak_excursion=float(out.max()),                        # beyond the window
        peak_dev=float(np.max(np.abs(Tp - T_TARGET))),          # from the set point
        t_out_pct=float(100.0 * left.mean()),
        left_window=int(bool(left.any())),
        cc_drop=float(np.max(np.abs(Cc[post] - Cc[shock_at - 1]))),
        settle_dev=(float(np.mean(np.abs(T[shock_at + 30:] - T_TARGET)))
                    if len(T) > shock_at + 30 else float(np.mean(np.abs(Tp - T_TARGET)))),
        recovery_steps=recovery,
        T_min=float(Tp.min()), T_max=float(Tp.max()),
    )


def learned_runs(pref_i):
    """Every frozen checkpoint for this preference, from the protocol selection."""
    out = []
    for s in tm.TRAIN_SEEDS:
        rd = os.path.join(OUT, "SAC", f"pref{pref_i}", f"seed{s}")
        sel = os.path.join(rd, "selection_protocol.json")
        if not os.path.exists(sel): continue
        j = json.load(open(sel))
        out.append((s, os.path.join(rd, f"ckpt_{j['ckpt']:06d}.pt")))
    if not out:
        raise SystemExit("no selection_protocol.json found; run reeval_protocol.py --stage select first")
    return out


def job(args):
    scen, name, kind, payload, pref_i = args
    w = tm.PREFERENCE_SET[pref_i]
    rows = []
    for seed in tm.TEST_SEEDS:
        env = SugarExtractionEnv(seed=seed, **UE.ENV_KW)
        ctrl = (FrozenPolicy("SAC", payload, w, seed=0) if kind == "learned"
                else build_classical(payload, env))
        m = run_shock(env, ctrl, scen, seed, w)
        rows.append(dict(scenario=scen, controller=name, seed=seed, **m))
    print(f"  {scen:9s} {name:14s} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref", type=int, default=2, help="preference index (2 = balanced)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    w = tm.PREFERENCE_SET[a.pref]
    runs = learned_runs(a.pref)
    print(f"Robustness under the Section 4.4 protocol: preference {w}, "
          f"{len(tm.TEST_SEEDS)} test seeds, {len(runs)} learned runs, {len(SCENARIOS)} scenarios")

    jobs = []
    for scen in SCENARIOS:
        for s, ck in runs:
            jobs.append((scen, f"MORL-SAC run{s}", "learned", ck, a.pref))
        for nm, cls in CLASSICAL:
            jobs.append((scen, nm, "classical", cls, a.pref))

    rows = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(job, jobs): rows += r
    else:
        for j in jobs: rows += job(j)
    tm.write_csv(os.path.join(OUT, "robustness_perseed.csv"), rows)

    # ---------------- aggregate, with the tail statistics ----------------
    def agg(sel):
        pk = np.array([r["peak_excursion"] for r in sel])
        dv = np.array([r["peak_dev"] for r in sel])
        rec = np.array([r["recovery_steps"] for r in sel], dtype=float)
        return dict(n=len(sel),
                    peak_mean=round(float(pk.mean()), 3),
                    peak_P95=round(float(np.percentile(pk, 95)), 3),
                    peak_worst=round(float(pk.max()), 3),
                    dev_mean=round(float(dv.mean()), 3),
                    dev_worst=round(float(dv.max()), 3),
                    t_out_pct=round(float(np.mean([r["t_out_pct"] for r in sel])), 2),
                    seeds_left_window=int(sum(r["left_window"] for r in sel)),
                    cc_drop_mean=round(float(np.mean([r["cc_drop"] for r in sel])), 3),
                    cc_drop_worst=round(float(np.max([r["cc_drop"] for r in sel])), 3),
                    settle_dev=round(float(np.mean([r["settle_dev"] for r in sel])), 3),
                    recovery_worst=(round(float(np.nanmax(rec)), 1) if not np.all(np.isnan(rec)) else "never left"))
    summ = []
    for scen in SCENARIOS:
        learned = [r for r in rows if r["scenario"] == scen and r["controller"].startswith("MORL")]
        summ.append(dict(scenario=scen, controller="MORL-SAC (all runs)", **agg(learned)))
        for s, _ in runs:
            sel = [r for r in rows if r["scenario"] == scen and r["controller"] == f"MORL-SAC run{s}"]
            summ.append(dict(scenario=scen, controller=f"MORL-SAC run{s}", **agg(sel)))
        for nm, _ in CLASSICAL:
            sel = [r for r in rows if r["scenario"] == scen and r["controller"] == nm]
            summ.append(dict(scenario=scen, controller=nm, **agg(sel)))
    tm.write_csv(os.path.join(OUT, "robustness_summary.csv"), summ)

    # ---------------- paired tests, learned vs each baseline ----------------
    from scipy.stats import wilcoxon
    tests = []
    for scen in SCENARIOS:
        for nm, _ in CLASSICAL:
            base = {r["seed"]: r for r in rows if r["scenario"] == scen and r["controller"] == nm}
            for s, _ in runs:
                lr = {r["seed"]: r for r in rows if r["scenario"] == scen and r["controller"] == f"MORL-SAC run{s}"}
                seeds = sorted(set(lr) & set(base))
                for metric in ("peak_excursion", "cc_drop"):
                    dl = np.array([lr[k][metric] - base[k][metric] for k in seeds])
                    p = (wilcoxon(dl, alternative="two-sided").pvalue
                         if not np.allclose(dl, 0) else 1.0)
                    tests.append(dict(scenario=scen, baseline=nm, learned_run=s, metric=metric,
                                      n_seeds=len(seeds), mean_diff=round(float(dl.mean()), 4),
                                      learned_worse_seeds=int((dl > 0).sum()), p=round(float(p), 4)))
    tm.write_csv(os.path.join(OUT, "robustness_tests.csv"), tests)

    print("\n" + "=" * 104)
    print(f"{'scenario':10s}{'controller':22s}{'peak mean':>11}{'P95':>8}{'worst':>8}"
          f"{'% t out':>9}{'seeds out':>11}{'Cc worst':>10}")
    print("-" * 104)
    for d in summ:
        if d["controller"].startswith("MORL-SAC run"): continue      # print the pooled row only
        print(f"{d['scenario']:10s}{d['controller']:22s}{d['peak_mean']:>11.3f}{d['peak_P95']:>8.3f}"
              f"{d['peak_worst']:>8.3f}{d['t_out_pct']:>9.2f}{d['seeds_left_window']:>8}/{d['n']:<3}"
              f"{d['cc_drop_worst']:>10.3f}")
    print("=" * 104)
    print("peak = worst excursion beyond the thermal window, degrees; a value of 0 means the")
    print("window was never left. Send robustness_summary.csv and robustness_tests.csv.")


if __name__ == "__main__":
    main()
