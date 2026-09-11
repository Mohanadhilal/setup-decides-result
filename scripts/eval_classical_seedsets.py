#!/usr/bin/env python3
"""
eval_classical_seedsets.py
==========================
Evaluate the classical controllers (PID, Fuzzy, optionally MPC) on the SAME
validation seeds (100-109) and test seeds (200-219) used for the learned
controllers in train_multiseed.py, under the same scalarization u_w.

Purpose (response to R2-2): the frozen learned checkpoints scored higher on
the unseen test seeds than on the validation seeds on all 15 runs. If the
classical controllers, which involve NO checkpoint selection, show the same
offset, the offset is a property of the two disturbance sets, not of
selection, and the reviewer's leakage concern is closed.

Outputs: classical_seedsets.csv  (controller, seed_set, seed, score, T12, ...)
         printed table of mean score per set and the offset (test - validation)

Run in the same folder as sugar_extraction_env.py, baseline_controllers.py,
compare_rl_algorithms.py, train_multiseed.py:
    python eval_classical_seedsets.py
"""
import csv, numpy as np
from train_multiseed import (VAL_SEEDS, TEST_SEEDS, PREFERENCE_SET, u_w, SS_WINDOW, T_LO, T_HI,
                             make_env, reset_env)
from baseline_controllers import PIDController, FuzzyController
try:
    from baseline_controllers import MPCController
    HAVE_MPC = True
except Exception:
    HAVE_MPC = False

PREF_I = 2                                     # balanced preference, as in the RL runs
W = PREFERENCE_SET[PREF_I]
INCLUDE_MPC = False                            # set True if you want MPC too (slower)


def make_controllers(env):
    c = {"PID":   PIDController(dt=env.dt),
         "Fuzzy": FuzzyController(dt=env.dt)}
    if INCLUDE_MPC and HAVE_MPC:
        c["MPC"] = MPCController(env.model, dt=env.dt)
    return c


def run_episode_classical(env, ctrl, seed):
    obs = reset_env(env, seed, W)
    ctrl.reset()
    # classical controllers need an initial info dict (as in compare_controllers.py)
    try:
        out0 = env.model.get_outputs(env.x)
        info = dict(out0, Kc=env.model.Kc, Pp=env.model.P_p, Gv=env.model.G_v)
    except Exception:
        info = {}
    q = None; T = []; last = None; done = False
    while not done:
        a = ctrl.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        rv = np.asarray(info["reward_vec"], dtype=float)
        q = rv if q is None else q + rv
        T.append(float(info["T12"])); last = info; done = term or trunc
    Tss = np.array(T[-SS_WINDOW:])
    return dict(score=u_w(q, W), T12=float(Tss.mean()), T_std=float(Tss.std()),
                T_max=float(np.max(T)), safe=bool(T_LO <= Tss.mean() <= T_HI),
                Cc=float(last["Cc"]), Gp=float(last["Gp"]), extraction=float(last["extraction"]))


def main():
    rows = []
    for set_name, seeds in (("validation", VAL_SEEDS), ("test", TEST_SEEDS)):
        for s in seeds:
            env = make_env(s, W)
            for name, ctrl in make_controllers(env).items():
                env = make_env(s, W)                       # fresh env per controller
                m = run_episode_classical(env, ctrl, s)
                rows.append(dict(controller=name, seed_set=set_name, seed=s, **m))
                print(f"{name:<6} {set_name:<10} seed={s:<4} score={m['score']:9.2f} T12={m['T12']:6.2f}", flush=True)
    with open("classical_seedsets.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("\n" + "=" * 72)
    print(f"{'controller':<10}{'validation':>14}{'test':>14}{'test - val':>14}{'safe(test)':>12}")
    print("-" * 72)
    names = sorted({r["controller"] for r in rows})
    for n in names:
        v = np.array([r["score"] for r in rows if r["controller"] == n and r["seed_set"] == "validation"])
        t = np.array([r["score"] for r in rows if r["controller"] == n and r["seed_set"] == "test"])
        sf = np.mean([r["safe"] for r in rows if r["controller"] == n and r["seed_set"] == "test"])
        print(f"{n:<10}{v.mean():>14.2f}{t.mean():>14.2f}{t.mean()-v.mean():>+14.2f}{sf:>12.2f}")
    print("=" * 72)
    print("Learned controllers (train_multiseed): test - val = SAC +32.5, TD3 +40.7, PPO +101.0")
    print("If the classical offsets are positive and of similar size, the offset is a")
    print("seed-set property, not selection bias.  Saved: classical_seedsets.csv")


if __name__ == "__main__":
    main()
