#!/usr/bin/env python3
"""
unified_evaluation.py
=====================
Produces Table 2 (controller comparison at the balanced preference) and
Table 3 (Pareto operating points of the fixed-weight policies) from a
SINGLE evaluation protocol, so that the balanced MORL-SAC row is
numerically identical in both tables.

Why this script exists
----------------------
compare_controllers.py and pareto_front.py used different protocols:
    seeds        1000..1004      vs   2000..2002
    episodes     5               vs   3
    aggregation  mean of last 30 vs   final instantaneous value
    preference   not passed to reset   passed to reset
Those four differences made the same policy report different numbers in
Table 2 and Table 3. Here every number comes from one protocol, one set
of seeds, and one checkpoint per preference.

Outputs (in runs/)
------------------
    table2_controllers.csv      Table 2 of the paper
    table3_pareto.csv           Table 3 of the paper
    unified_eval_raw.csv        per-episode values (for statistics)
    unified_eval_report.txt     printed tables + consistency check

Run in PyCharm:  Run 'unified_evaluation'
"""

import os
import numpy as np

from sugar_extraction_env import SugarExtractionEnv
from baseline_controllers import (PIDController, MPCController,
                                  FuzzyController, MORLAgent)
from baseline_controllers_mv import (PIDControllerMV as PIDController,
                                     MPCControllerMV as MPCController,
                                     FuzzyControllerMV as FuzzyController)

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ============================================================
# ONE protocol for everything. Change here and both tables follow.
# ============================================================
N_EPISODES = 10            # episodes per configuration
SEED_BASE = 1000          # seeds are SEED_BASE .. SEED_BASE+N_EPISODES-1
AVG_WINDOW = 30           # steps averaged at the end of the episode
STD_WINDOW = 50           # steps used for the temperature standard deviation
T_TARGET = 73.5           # thermal set point used for the tracking error
SETTLE_SKIP = 120          # initial steps ignored when scoring safety


def get_safety_window():
    """Read the thermal safety window from the environment, so this script
    can never disagree with the reward function.

    Order of preference:
      1. module-level constants T_LO / T_HI in sugar_extraction_env.py
      2. the literal 'T_lo, T_hi = a, b' inside the reward function
      3. a hard-coded fallback

    Returns (lo, hi, source_description).
    """
    import sugar_extraction_env as _envmod

    lo = getattr(_envmod, "T_LO", None)
    hi = getattr(_envmod, "T_HI", None)
    if lo is not None and hi is not None:
        return float(lo), float(hi), "module constants"

    try:
        import inspect
        import re
        src = inspect.getsource(_envmod)
        m = re.search(r"T_lo\s*,\s*T_hi\s*=\s*([0-9.]+)\s*,\s*([0-9.]+)", src)
        if m:
            return (float(m.group(1)), float(m.group(2)),
                    "parsed from the reward function")
    except Exception:
        pass

    return 70.0, 78.0, ("FALLBACK, could not read the environment; verify "
                        "this matches the reward function")


T_LO, T_HI, T_WINDOW_SOURCE = get_safety_window()

ENV_KW = dict(enable_disturbance=True, domain_randomize=False)

# Must match PREFERENCE_SET in train_morl_sac.py
PREFERENCE_SET = [
    [0.7, 0.1, 0.2],    # extraction priority
    [0.5, 0.3, 0.2],
    [0.34, 0.33, 0.33], # balanced
    [0.2, 0.6, 0.2],    # energy priority
    [0.2, 0.2, 0.6],    # concentration priority
]
BALANCED_INDEX = 2                      # index of the balanced preference
BALANCED_W = PREFERENCE_SET[BALANCED_INDEX]
BALANCED_CKPT = f"runs/actor_fixed{BALANCED_INDEX}.pt"


# ============================================================
def run_episode(env, controller, seed, preference):
    """One episode under the unified protocol. Returns scalar metrics."""
    obs, _ = env.reset(seed=seed, options={"preference": preference})
    controller.reset() if hasattr(controller, "reset") else None

    info = dict(env.model.get_outputs(env.x), Kc=env.model.Kc)
    T12, Cc, Gp = [], [], []

    done = False
    while not done:
        action = controller.act(obs, info)
        obs, _, term, trunc, info = env.step(action)
        T12.append(info["T12"])
        Cc.append(info["Cc"])
        Gp.append(info["Gp"])
        done = term or trunc

    T12 = np.asarray(T12); Cc = np.asarray(Cc); Gp = np.asarray(Gp)

    # ---- safety metrics -------------------------------------------------
    # Scored after the start-up transient, since the initial approach to the
    # operating point is not a controller failure.
    Tsafe = T12[SETTLE_SKIP:] if len(T12) > SETTLE_SKIP else T12
    below = np.maximum(T_LO - Tsafe, 0.0)
    above = np.maximum(Tsafe - T_HI, 0.0)
    violation = np.maximum(below, above)          # 0 inside the window
    in_window = float(np.mean(violation == 0.0)) * 100.0

    return {
        "Cc":         float(np.mean(Cc[-AVG_WINDOW:])),
        "Gp":         float(np.mean(Gp[-AVG_WINDOW:])),
        "T12":        float(np.mean(T12[-AVG_WINDOW:])),
        "T12_std":    float(np.std(T12[-STD_WINDOW:])),
        "T12_err":    float(np.mean(np.abs(T12[-STD_WINDOW:] - T_TARGET))),
        "extraction": float(info["extraction"]) * 100.0,
        # safety
        "in_window":  in_window,                       # % of steps inside
        # Average of the per-step safety penalty. The environment
        # charges the penalty at every step, so scoring must use the
        # mean of the penalty, never the penalty of the mean: the
        # penalty is convex, so the latter systematically forgives
        # exactly the oscillating policies that deserve charging.
        "mean_penalty": float(np.mean(
            (1.0 * violation + 0.5 * violation ** 2)[-AVG_WINDOW:])),
        "max_viol":   float(np.max(violation)),        # worst excursion, C
        "mean_viol":  float(np.mean(violation)),       # average excursion, C
        "T12_min":    float(np.min(Tsafe)),
        "T12_max":    float(np.max(Tsafe)),
    }


def evaluate(make_controller, label, preference):
    """Run one configuration over the shared seed list."""
    print(f"  evaluating {label} ...")
    per_ep = []
    for ep in range(N_EPISODES):
        seed = SEED_BASE + ep
        env = SugarExtractionEnv(seed=seed, **ENV_KW)
        controller = make_controller(env)
        per_ep.append(run_episode(env, controller, seed, preference))

    keys = per_ep[0].keys()
    mean = {k: float(np.mean([e[k] for e in per_ep])) for k in keys}
    std = {k: float(np.std([e[k] for e in per_ep])) for k in keys}
    # Safety is reported worst-case across seeds: a controller is only as safe
    # as its worst episode, so averaging the violation would hide it.
    worst = {
        "max_viol": float(np.max([e["max_viol"] for e in per_ep])),
        "in_window": float(np.min([e["in_window"] for e in per_ep])),
        "n_bad_seeds": int(sum(1 for e in per_ep if e["max_viol"] > 0.0)),
        "T12_min": float(np.min([e["T12_min"] for e in per_ep])),
        "T12_max": float(np.max([e["T12_max"] for e in per_ep])),
    }
    return {"label": label, "mean": mean, "std": std,
            "worst": worst, "episodes": per_ep}


# ============================================================
def main():
    os.makedirs("runs", exist_ok=True)

    if not os.path.exists(BALANCED_CKPT):
        raise SystemExit(
            f"Missing {BALANCED_CKPT}. Train the fixed-weight policies first "
            f"(train_morl_sac.py) before running this script."
        )

    print("=" * 74)
    print("Unified evaluation")
    print(f"  seeds        : {SEED_BASE}..{SEED_BASE + N_EPISODES - 1}")
    print(f"  episodes     : {N_EPISODES}")
    print(f"  aggregation  : mean of last {AVG_WINDOW} steps "
          f"(std over last {STD_WINDOW})")
    print(f"  disturbances : on, domain randomization off")
    print(f"  safety window: [{T_LO:.0f}, {T_HI:.0f}] C "
          f"({T_WINDOW_SOURCE})")
    print("=" * 74)

    # ---------- Table 2: controllers at the balanced preference ----------
    print("\n[Table 2] controllers at the balanced preference "
          f"w = {BALANCED_W}")

    results_t2 = []
    results_t2.append(evaluate(lambda env: PIDController(dt=env.dt),
                               "PID", BALANCED_W))
    results_t2.append(evaluate(lambda env: MPCController(env.model, dt=env.dt),
                               "MPC", BALANCED_W))
    results_t2.append(evaluate(lambda env: FuzzyController(dt=env.dt),
                               "Fuzzy", BALANCED_W))

    morl_balanced = evaluate(
        lambda env: MORLAgent(BALANCED_CKPT, preference=BALANCED_W),
        "MORL-SAC", BALANCED_W)
    results_t2.append(morl_balanced)

    # ---------- Table 3: fixed-weight policies across preferences ----------
    print("\n[Table 3] fixed-weight policies across the preference set")

    results_t3 = []
    for i, w in enumerate(PREFERENCE_SET):
        # Reuse the already-computed balanced run so the two tables cannot
        # disagree: same checkpoint, same seeds, same numbers.
        if i == BALANCED_INDEX:
            r = dict(morl_balanced)
            r["label"] = f"w={w} (balanced)"
            r["w"] = w
            results_t3.append(r)
            print(f"  reusing balanced run from Table 2 ({BALANCED_CKPT})")
            continue

        ckpt = f"runs/actor_fixed{i}.pt"
        if not os.path.exists(ckpt):
            print(f"  (missing {ckpt}, skipping)")
            continue
        r = evaluate(lambda env, c=ckpt, ww=w: MORLAgent(c, preference=ww),
                     f"w={w}", w)
        r["w"] = w
        results_t3.append(r)

    # ---------- print ----------
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("\n" + "=" * 116)
    emit("Table 2. Steady-state performance at the balanced preference "
         "(disturbances active).")
    emit("-" * 116)
    emit(f"{'Controller':<13}{'Cc, %':<15}{'Gp, kg/s':<16}{'T12, C':<15}"
         f"{'std(T12)':<11}{'Extr, %':<10}{'In-window, %':<14}{'Max viol, C':<12}")
    emit("-" * 116)
    for r in results_t2:
        m, s, w = r["mean"], r["std"], r["worst"]
        emit(f"{r['label']:<13}"
             f"{m['Cc']:.2f}+-{s['Cc']:.2f}    "
             f"{m['Gp']:.3f}+-{s['Gp']:.3f}   "
             f"{m['T12']:.2f}+-{s['T12']:.2f}  "
             f"{m['T12_std']:.3f}      "
             f"{m['extraction']:.2f}     "
             f"{w['in_window']:.1f}          "
             f"{w['max_viol']:.3f}")
    emit("-" * 116)
    emit("In-window and Max viol are worst case over the seeds; the safe "
         f"window is [{T_LO:.0f}, {T_HI:.0f}] C.")
    emit("=" * 116)

    emit("\n" + "=" * 116)
    emit("Table 3. Operating points of the fixed-weight policies.")
    emit("-" * 116)
    emit(f"{'Preference':<24}{'Cc, %':<10}{'Gp, kg/s':<11}{'T12, C':<10}"
         f"{'Extr, %':<10}{'std(T12)':<11}{'In-window, %':<14}{'Max viol, C':<12}")
    emit("-" * 116)
    for r in results_t3:
        m, w = r["mean"], r["worst"]
        emit(f"{str(r['w']):<24}"
             f"{m['Cc']:.2f}     "
             f"{m['Gp']:.3f}      "
             f"{m['T12']:.2f}     "
             f"{m['extraction']:.2f}     "
             f"{m['T12_std']:.3f}      "
             f"{w['in_window']:.1f}          "
             f"{w['max_viol']:.3f}")
    emit("=" * 116)

    # ---------- dedicated safety audit ----------
    emit("\n" + "=" * 116)
    emit(f"Safety audit. Window [{T_LO:.0f}, {T_HI:.0f}] C, scored after the "
         f"first {SETTLE_SKIP} steps, over {N_EPISODES} seeds.")
    emit("-" * 116)
    emit(f"{'Configuration':<26}{'In-window, %':<15}{'Max viol, C':<14}"
         f"{'Seeds violating':<18}{'T12 min':<11}{'T12 max':<11}{'Verdict':<10}")
    emit("-" * 116)
    all_rows = ([(r["label"], r) for r in results_t2]
                + [(f"MORL w={r['w']}", r) for r in results_t3
                   if r.get("w") != BALANCED_W])
    any_violation = False
    for label, r in all_rows:
        w = r["worst"]
        bad = w["max_viol"] > 0.0
        any_violation |= bad
        emit(f"{label:<26}"
             f"{w['in_window']:.1f}           "
             f"{w['max_viol']:.3f}         "
             f"{w['n_bad_seeds']}/{N_EPISODES}               "
             f"{w['T12_min']:.2f}      "
             f"{w['T12_max']:.2f}      "
             f"{'VIOLATES' if bad else 'safe'}")
    emit("-" * 116)
    emit("  => " + ("Every configuration stayed inside the safe window."
                    if not any_violation else
                    "At least one configuration left the safe window. Report "
                    "this explicitly in the paper or strengthen the penalty."))
    emit("=" * 116)

    # ---------- consistency check ----------
    emit("\nConsistency check (Table 2 MORL-SAC row vs Table 3 balanced row)")
    bal_t3 = next(r for r in results_t3 if r.get("w") == BALANCED_W)
    ok = True
    for k in ["Cc", "Gp", "T12", "extraction", "T12_std"]:
        a = morl_balanced["mean"][k]
        b = bal_t3["mean"][k]
        same = abs(a - b) < 1e-9
        ok &= same
        emit(f"  {k:<12} Table2={a:.6f}  Table3={b:.6f}  "
             f"{'identical' if same else 'MISMATCH'}")
    emit(f"  => {'PASS: the two tables agree exactly.' if ok else 'FAIL'}")

    # ---------- paired statistics vs baselines ----------
    if HAVE_SCIPY:
        emit("\nPaired Wilcoxon signed-rank tests, MORL-SAC vs each baseline")
        emit("(same seeds, two-sided; n = %d episodes)" % N_EPISODES)
        for r in results_t2[:3]:
            for k in ["Gp", "extraction", "T12_std"]:
                x = [e[k] for e in morl_balanced["episodes"]]
                y = [e[k] for e in r["episodes"]]
                try:
                    stat, p = wilcoxon(x, y)
                    emit(f"  {r['label']:<6} {k:<12} p = {p:.4f}")
                except Exception as exc:
                    emit(f"  {r['label']:<6} {k:<12} n/a ({exc})")
        emit("  Note: with 5 seeds the smallest attainable two-sided p is "
             "0.0625, so treat these as indicative. Increase N_EPISODES to "
             "10 or more for a claim of significance.")
    else:
        emit("\n(scipy not installed: skipping Wilcoxon tests. "
             "pip install scipy)")

    # ---------- save ----------
    with open("runs/table2_controllers.csv", "w") as f:
        f.write("Controller,Cc_mean,Cc_std,Gp_mean,Gp_std,T12_mean,T12_std_across_seeds,"
                "T12_ripple,T12_target_error,Extraction,"
                "InWindow_pct_worst,MaxViolation_C_worst,MeanViolation_C,"
                "SeedsViolating,T12_min,T12_max\n")
        for r in results_t2:
            m, s, w = r["mean"], r["std"], r["worst"]
            f.write(f"{r['label']},{m['Cc']:.4f},{s['Cc']:.4f},"
                    f"{m['Gp']:.4f},{s['Gp']:.4f},{m['T12']:.4f},{s['T12']:.4f},"
                    f"{m['T12_std']:.4f},{m['T12_err']:.4f},{m['extraction']:.4f},"
                    f"{w['in_window']:.2f},{w['max_viol']:.4f},{m['mean_viol']:.4f},"
                    f"{w['n_bad_seeds']},{w['T12_min']:.3f},{w['T12_max']:.3f}\n")

    with open("runs/table3_pareto.csv", "w") as f:
        f.write("w_ext,w_energy,w_conc,Cc,Gp,T12,Extraction,T12_ripple,"
                "InWindow_pct_worst,MaxViolation_C_worst,SeedsViolating\n")
        for r in results_t3:
            m, ww, sw = r["mean"], r["w"], r["worst"]
            f.write(f"{ww[0]},{ww[1]},{ww[2]},{m['Cc']:.4f},{m['Gp']:.4f},"
                    f"{m['T12']:.4f},{m['extraction']:.4f},{m['T12_std']:.4f},"
                    f"{sw['in_window']:.2f},{sw['max_viol']:.4f},"
                    f"{sw['n_bad_seeds']}\n")

    with open("runs/unified_eval_raw.csv", "w") as f:
        f.write("config,seed,Cc,Gp,T12,T12_ripple,T12_target_error,Extraction,"
                "InWindow_pct,MaxViolation_C,T12_min,T12_max\n")
        def dump(rows):
            for r in rows:
                for ep, e in enumerate(r["episodes"]):
                    f.write(f"\"{r['label']}\",{SEED_BASE + ep},{e['Cc']:.4f},"
                            f"{e['Gp']:.4f},{e['T12']:.4f},{e['T12_std']:.4f},"
                            f"{e['T12_err']:.4f},{e['extraction']:.4f},"
                            f"{e['in_window']:.2f},{e['max_viol']:.4f},"
                            f"{e['T12_min']:.3f},{e['T12_max']:.3f}\n")
        dump(results_t2)
        dump([r for r in results_t3 if r.get("w") != BALANCED_W])

    with open("runs/unified_eval_report.txt", "w") as f:
        f.write("\n".join(lines))

    print("\nSaved:")
    print("  runs/table2_controllers.csv")
    print("  runs/table3_pareto.csv")
    print("  runs/unified_eval_raw.csv")
    print("  runs/unified_eval_report.txt")


if __name__ == "__main__":
    main()
