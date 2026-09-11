#!/usr/bin/env python3
"""
classical_grid_sweep.py
=======================
Sweeps both baseline set points, temperature and concentration, and draws the
front a classical cascade can actually reach.

Why the previous sweep was in the wrong dimension
-------------------------------------------------
classical_front_sweep.py varied the concentration set point and held the
temperature set point at 73.5 degrees for every controller. Section 3.4
establishes that in this model extraction is a function of temperature alone
over the operating range, so varying concentration moves the energy axis and
leaves extraction fixed. The resulting classical front is close to a
horizontal line, which is not a trade-off curve at all, and the headline
comparison against it reduced to the learned controller having settled 0.54
degrees cooler than a set point that was never swept.

That makes the earlier comparison uninformative rather than wrong. A cascade
with a temperature loop and a concentration loop has two degrees of freedom
on the two steady-state variables that determine all three objectives, so
sweeping only one of them explores a line through a two-dimensional set.

What this does
--------------
Evaluates every combination of temperature and concentration set point on a
grid, under the same protocol, seeds and safety accounting as everywhere
else, and reports the non-dominated classical front in the plane of
extraction against plant-level steam. It then compares that front with the
learned one and states which dominates where.

What to expect, and why it still matters
----------------------------------------
Given Section 3.4, the classical cascade should now trace a genuine curve and
should be able to reach most of the attainable set. If it dominates the
learned front outright, that is the structural result argued in Section 5.3
and it should be reported as such: on a plant whose objectives are fixed by
two steady-state variables, a two-loop cascade is hard to beat at steady
state, and the case for a learned controller has to be made on transients,
constraint handling, or preference-to-set-point translation instead. The
point of running it is to replace an argument with a measurement.

Run:  python classical_grid_sweep.py
Writes runs/classical_grid.csv and runs/classical_grid.png
"""

import os
import numpy as np

from sugar_extraction_env import SugarExtractionEnv
from baseline_controllers_mv import (PIDControllerMV, MPCControllerMV,
                                     FuzzyControllerMV)
from unified_evaluation import run_episode, N_EPISODES, SEED_BASE, ENV_KW
import morl_score as MS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# ============================================================
# Temperature set points span the thermal window with a margin, since the
# window is a hard constraint and a set point on the boundary would fail.
T_TARGETS = [71.0, 72.0, 73.0, 73.5, 74.5, 75.5, 76.5, 77.0]
CC_TARGETS = [11.80, 12.10, 12.30, 12.50]
SCREEN_SEEDS = 3
RIPPLE_LIMIT = 0.15

LEARNED = [(2.6643, 97.7633), (2.5993, 97.5725), (2.3550, 97.1451),
           (1.9268, 96.3322), (2.3971, 96.9732)]
GAP_POLICY = (2.1704, 96.8094)
# ============================================================

CTRLS = [("PID", PIDControllerMV), ("MPC", MPCControllerMV),
         ("Fuzzy", FuzzyControllerMV)]
PREF = [0.34, 0.33, 0.33]      # only to satisfy the environment interface


def build(cls, env, T, cc):
    if cls is MPCControllerMV:
        return cls(env.model, T_target=T, Cc_target=cc, dt=env.dt)
    return cls(T_target=T, Cc_target=cc, dt=env.dt)


def run(cls, T, cc, n_seeds):
    vals = []
    for k in range(n_seeds):
        seed = SEED_BASE + k
        env = SugarExtractionEnv(seed=seed, **ENV_KW)
        vals.append(run_episode(env, build(cls, env, T, cc), seed, PREF))
    m = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return m, max(v["max_viol"] for v in vals), \
        float(np.mean([v["T12_std"] for v in vals]))


def staircase(pts):
    xs, ys = [], []
    for x, y in sorted(pts):
        if not ys or y > ys[-1]:
            xs.append(x); ys.append(y)
    return xs, ys


def main():
    os.makedirs("runs", exist_ok=True)
    n_pts = len(T_TARGETS) * len(CC_TARGETS) * len(CTRLS)
    print("=" * 96)
    print("Two-dimensional baseline set-point sweep")
    print(f"  temperature set points : {T_TARGETS}")
    print(f"  concentration set points: {CC_TARGETS}")
    print(f"  evaluations            : {n_pts} at {SCREEN_SEEDS} seeds")
    print("=" * 96)

    rows = []
    for name, cls in CTRLS:
        for T in T_TARGETS:
            for cc in CC_TARGETS:
                m, viol, rip = run(cls, T, cc, SCREEN_SEEDS)
                rows.append(dict(ctrl=name, T_set=T, cc_set=cc, mean=m,
                                 viol=viol, ripple=rip,
                                 gt=MS.total_steam(m["Gp"], m["extraction"],
                                                   m["Cc"])))
        print(f"  {name} done")

    ok = [r for r in rows if r["viol"] <= 0 and r["ripple"] <= RIPPLE_LIMIT]
    print(f"\n  admissible points: {len(ok)} of {len(rows)}")
    if not ok:
        raise SystemExit("No admissible classical point.")

    ex = np.array([r["mean"]["extraction"] for r in ok])
    gt = np.array([r["gt"] for r in ok])
    T12 = np.array([r["mean"]["T12"] for r in ok])
    print(f"  extraction spanned : {ex.min():.3f} to {ex.max():.3f} "
          f"(range {np.ptp(ex):.3f})")
    print(f"  total steam spanned: {gt.min():.4f} to {gt.max():.4f} "
          f"(range {np.ptp(gt):.4f})")
    print(f"  previous one-dimensional sweep spanned 0.171 in steam and "
          f"0.35 in extraction")

    # does temperature explain extraction, as Section 3.4 implies
    c = np.corrcoef(T12, ex)[0, 1]
    print(f"  correlation of extraction with settled temperature: {c:+.4f}")

    cx, cy = staircase(list(zip(gt, ex)))
    lx, ly = staircase(LEARNED)
    print(f"\n  classical front points: {len(cx)}")

    lo, hi = max(min(lx), min(cx)), min(max(lx), max(cx))
    print("\n" + "=" * 96)
    if hi <= lo:
        print("  The two fronts do not overlap in total steam.")
    else:
        grid = np.linspace(lo, hi, 40)
        d = np.interp(grid, lx, ly) - np.interp(grid, cx, cy)
        print(f"  overlap {lo:.4f} to {hi:.4f} kg/s")
        print(f"  learned minus classical: mean {d.mean():+.4f}, "
              f"min {d.min():+.4f}, max {d.max():+.4f}")
        print(f"  learned ahead over {float(np.mean(d > 0))*100:.0f}% "
              f"of the overlap")
        if d.max() < -0.02:
            v = ("The classical cascade dominates across the overlap. With "
                 "both set points swept this is the structural result of "
                 "Section 5.3, now measured rather than argued.")
        elif d.min() > 0.02:
            v = ("The learned front dominates. The earlier negative result "
                 "was an artefact of sweeping only one set point.")
        else:
            v = ("The fronts are close. Neither dominates, and the honest "
                 "claim is that a two-loop cascade and a preference-driven "
                 "controller reach comparable steady-state performance here.")
        print(f"\n  VERDICT: {v}")

        # where does the gap-filling policy sit now
        cls_at = float(np.interp(GAP_POLICY[0], cx, cy))
        print(f"\n  gap-filling policy at {GAP_POLICY[0]:.4f}: "
              f"{GAP_POLICY[1]:.4f} against classical {cls_at:.4f} "
              f"({GAP_POLICY[1]-cls_at:+.4f})")
        print(f"  the one-dimensional sweep gave -0.104 at this point")
    print("=" * 96)

    with open("runs/classical_grid.csv", "w") as f:
        f.write("controller,T_set,Cc_set,Cc,Gp,G_total,T12,Extraction,"
                "ripple,max_violation,admissible\n")
        for r in rows:
            m = r["mean"]
            adm = int(r["viol"] <= 0 and r["ripple"] <= RIPPLE_LIMIT)
            f.write(f"{r['ctrl']},{r['T_set']},{r['cc_set']},{m['Cc']:.4f},"
                    f"{m['Gp']:.4f},{r['gt']:.4f},{m['T12']:.4f},"
                    f"{m['extraction']:.4f},{r['ripple']:.4f},"
                    f"{r['viol']:.4f},{adm}\n")

    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        sc = ax.scatter(gt, ex, c=T12, s=42, cmap="coolwarm", zorder=3,
                        label="classical cascade, both set points swept")
        plt.colorbar(sc, ax=ax, label="settled temperature, °C")
        ax.plot(cx, cy, "-", color="0.35", lw=1.5, zorder=4,
                label="classical front")
        ax.plot(lx, ly, "-o", color="tab:blue", lw=1.8, ms=6, zorder=5,
                label="learned front")
        ax.scatter([GAP_POLICY[0]], [GAP_POLICY[1]], marker="*", s=150,
                   color="tab:red", edgecolors="k", linewidths=.6, zorder=7,
                   label="learned policy trained for the band")
        ax.set_xlabel("total steam, diffuser + evaporation, kg/s")
        ax.set_ylabel("extraction efficiency, %")
        ax.set_title("Classical front with both set points swept,\n"
                     "against the learned front")
        ax.grid(alpha=.3); ax.legend(fontsize=8.4, loc="lower right")
        fig.tight_layout(); fig.savefig("runs/classical_grid.png", dpi=165)
        print("\nSaved runs/classical_grid.png")
    print("Saved runs/classical_grid.csv")


if __name__ == "__main__":
    main()
