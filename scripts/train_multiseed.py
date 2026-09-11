#!/usr/bin/env python3
"""
train_multiseed.py
==================
Multi-seed training + seed-separated evaluation for the IJIES revision.
Implements Protocol D (reviewer R2-2) exactly:

    TRAIN seeds       0..4       -> policy init + env randomness during training only
    VALIDATION seeds  100..109   -> checkpoint scoring + safe-survivor selection ONLY
    TEST seeds        200..219   -> every reported number, evaluated ONCE on the frozen pick

For each (algorithm x preference x train-seed):
    1. train, saving a checkpoint every CKPT_EVERY steps
    2. score every checkpoint on VALIDATION seeds under u_w; apply the thermal gate
    3. select the highest-scoring SAFE checkpoint -> freeze in selection.json
    4. evaluate the frozen checkpoint once on TEST seeds -> test.csv
    5. record selection bias = validation score - test score

Outputs (all under OUT_DIR):
    <algo>/<pref_i>/<seed_k>/ckpt_XXXXXX.pt, validation.csv, selection.json, test.csv, DONE
    results.csv        one row per (algo, pref, seed)
    summary.csv        mean +- std across the five train seeds, per (algo, pref)
    selection_bias.csv the table that answers R2-2 directly

Run:
    python train_multiseed.py --workers 5            # real runs, 5 in parallel
    python train_multiseed.py --mock                 # 30-second end-to-end self-test
    python train_multiseed.py --algos SAC --prefs 2  # subset

WIRE-UP: edit the ADAPTER block only. Nothing else needs to change.
"""

import os, sys, json, csv, time, argparse, itertools
import numpy as np

# ============================================================================
#                                  CONFIG
# ============================================================================
ALGOS          = ["SAC", "TD3", "PPO"]
TRAIN_SEEDS    = [0, 1, 2, 3, 4]
VAL_SEEDS      = list(range(100, 110))     # 10
TEST_SEEDS     = list(range(200, 220))     # 20

# Objectives: w = [extraction, energy, concentration]  (must match the paper)
PREFERENCE_SET = [
    [0.7, 0.1, 0.2],
    [0.5, 0.3, 0.2],
    [0.34, 0.33, 0.33],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6],
    [0.29, 0.51, 0.20],   # index 5: gap-fill preference (classical energy band), headline_statistics.py
]

STEPS        = 150_000     # identical for all algorithms (fair comparison)
WARMUP       = 5_000
CKPT_EVERY   = 10_000      # 15 checkpoints per run
EVAL_EPISODES_PER_SEED = 1

SCALARIZATION = "linear"   # "linear" | "tchebycheff"
Z_STAR        = None       # ideal point [z1,z2,z3] for Tchebycheff; None -> linear
RHO           = 0.05       # Tchebycheff augmentation

T_LO, T_HI   = 70.0, 78.0  # thermal safety window (steady-state gate)
SS_WINDOW    = 50          # last N steps define steady state

OUT_DIR      = "runs_multiseed"
TORCH_THREADS_PER_WORKER = 1

# ============================================================================
#                                  ADAPTER
#   Wired to compare_rl_algorithms.py (verified against the real classes).
#   Three mismatches are handled here:
#     1. constructors take (o, a, h=, lr=, gamma=, tau=) and no seed/batch
#     2. update() takes a replay buffer, not a step; PPO is on-policy
#     3. the classes have no save()/load(), so checkpointing is added here
# ============================================================================
def make_env(seed, preference, training=True):
    """training=True -> domain randomization on (robustness); False -> nominal plant, the unified evaluation protocol."""
    from sugar_extraction_env import SugarExtractionEnv
    return SugarExtractionEnv(seed=seed, enable_disturbance=True, domain_randomize=training)


def reset_env(env, seed, preference):
    obs, _ = env.reset(seed=seed, options={"preference": preference})
    return obs


def make_agent(algo, obs_dim, act_dim, seed, preference=None):
    """Return an object exposing act/observe/update/save/load for the orchestrator.
    preference: the weight vector w; every stored training reward is u_w(reward_vec, w),
    NOT the environment's default scalar (which ignores w)."""
    import torch, numpy as np
    from compare_rl_algorithms import SAC, TD3, PPO, ReplayBuffer

    torch.manual_seed(seed)            # the classes take no seed; set it here
    np.random.seed(seed)

    H, LR, GAMMA, TAU, BATCH = 256, 2e-4, 0.99, 0.005, 256

    if algo == "SAC":
        inner = SAC(obs_dim, act_dim, h=H, lr=LR, gamma=GAMMA, tau=TAU)
    elif algo == "TD3":
        inner = TD3(obs_dim, act_dim, h=H, lr=LR, gamma=GAMMA, tau=TAU)
    elif algo == "PPO":
        inner = PPO(obs_dim, act_dim, h=H, lr=LR, gamma=GAMMA)   # no tau for PPO
    else:
        raise ValueError(f"unknown algo {algo}")

    off_policy = algo in ("SAC", "TD3")

    class Wrap:
        name = algo

        def __init__(s):
            s.inner = inner
            s.w = None if preference is None else np.asarray(preference, dtype=float)
            s.buf = ReplayBuffer(300_000, obs_dim, act_dim) if off_policy else None
            s.roll = {k: [] for k in ("obs", "act", "rew", "done", "val", "logp")}

        # -- action -------------------------------------------------------
        def act(s, obs, deterministic=False):
            return s.inner.act(obs, deterministic=deterministic)

        # -- transition ---------------------------------------------------
        def observe(s, o, a, r, r_vec, o2, done):
            # PREFERENCE-SCALARIZED training reward (this is the fix): u_w = w . r[0:3] + r[3]
            r_train = float(u_w(r_vec, s.w)) if s.w is not None else float(r)
            if off_policy:
                s.buf.add(o, a, r_train, o2, float(done))
            else:
                # PPO: act() stored (logp, value) in inner._last on the last call
                logp, v = getattr(s.inner, "_last", (0.0, 0.0))
                s.roll["obs"].append(o);   s.roll["act"].append(a)
                s.roll["rew"].append(r_train); s.roll["done"].append(float(done))
                s.roll["val"].append(v);   s.roll["logp"].append(logp)

        # -- learning -----------------------------------------------------
        def update(s, step):
            if off_policy:
                s.inner.update(s.buf, BATCH)
            elif len(s.roll["obs"]) >= s.inner.rollout:
                s.inner.update_from_rollout(s.roll["obs"], s.roll["act"], s.roll["rew"],
                                            s.roll["done"], s.roll["val"], s.roll["logp"])
                for k in s.roll: s.roll[k] = []

        # -- checkpointing (not present in the original classes) ----------
        def _modules(s):
            names = (["actor", "q1", "q2", "q1t", "q2t"] if algo == "SAC" else
                     ["actor", "actor_t", "q1", "q2", "q1t", "q2t"] if algo == "TD3" else
                     ["ac"])
            return {n: getattr(s.inner, n) for n in names if hasattr(s.inner, n)}

        def save(s, path):
            blob = {n: m.state_dict() for n, m in s._modules().items()}
            if hasattr(s.inner, "log_alpha"):
                blob["log_alpha"] = s.inner.log_alpha.detach().cpu()
            torch.save(blob, path)

        def load(s, path):
            blob = torch.load(path, map_location="cpu")
            for n, m in s._modules().items():
                if n in blob: m.load_state_dict(blob[n])
            if "log_alpha" in blob and hasattr(s.inner, "log_alpha"):
                with torch.no_grad():
                    s.inner.log_alpha.copy_(blob["log_alpha"].to(s.inner.log_alpha.device))

    return Wrap()
# ============================================================================


# ---------------------------------------------------------------- utilities
def u_w(q, w):
    """Scalarize a vector return q. Paper form: U(q_{1:3}; w) + q_4 (penalty unweighted)."""
    q = np.asarray(q, dtype=float); w = np.asarray(w, dtype=float)
    q3, pen = q[:3], (q[3] if q.shape[0] >= 4 else 0.0)
    if SCALARIZATION == "tchebycheff" and Z_STAR is not None:
        z = np.asarray(Z_STAR, dtype=float)
        U = -np.max(w * (z - q3)) + RHO * np.sum(w * q3)
    else:
        U = float(np.sum(w * q3))
    return float(U + pen)

def run_episode(env, agent, seed, w, deterministic=True):
    """One evaluation episode. Returns dict of episode metrics."""
    obs = reset_env(env, seed, w)
    q = None; T = []; last = None; done = False
    while not done:
        a = agent.act(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(a)
        rv = np.asarray(info["reward_vec"], dtype=float)
        q = rv if q is None else q + rv
        T.append(float(info["T12"])); last = info; done = term or trunc
    Tss = np.array(T[-SS_WINDOW:])
    return dict(
        score=u_w(q, w),
        T12=float(Tss.mean()), T_std=float(Tss.std()),
        T_min=float(np.min(T)), T_max=float(np.max(T)),
        safe=bool((Tss.mean() >= T_LO) and (Tss.mean() <= T_HI)),
        Cc=float(last["Cc"]), Gp=float(last["Gp"]),
        extraction=float(last["extraction"]),
    )

def evaluate(agent, w, seeds):
    rows = []
    for s in seeds:
        env = make_env(s, w, training=False)          # evaluation on the nominal plant
        for k in range(EVAL_EPISODES_PER_SEED):
            rows.append(dict(seed=s, **run_episode(env, agent, s + 1000 * k, w)))
    keys = ["score", "T12", "T_std", "T_min", "T_max", "Cc", "Gp", "extraction"]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    agg["score_std"] = float(np.std([r["score"] for r in rows]))
    agg["safe_all"] = bool(all(r["safe"] for r in rows))
    agg["safe_frac"] = float(np.mean([r["safe"] for r in rows]))
    return agg, rows

def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)


# ---------------------------------------------------------------- one run
def run_one(job):
    algo, pref_i, seed, resume = job
    w = PREFERENCE_SET[pref_i]
    rd = os.path.join(OUT_DIR, algo, f"pref{pref_i}", f"seed{seed}")
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "DONE")):
        return f"skip {rd}"
    try:
        import torch; torch.set_num_threads(TORCH_THREADS_PER_WORKER); torch.manual_seed(seed)
    except Exception: pass
    np.random.seed(seed)

    # ---- 1. train ----
    env = make_env(seed, w)
    obs = reset_env(env, seed, w)
    obs_dim, act_dim = obs.shape[0], env.action_space.shape[0]
    agent = make_agent(algo, obs_dim, act_dim, seed, w)

    ckpts = sorted(int(f[5:11]) for f in os.listdir(rd) if f.startswith("ckpt_"))
    start = 0
    if resume and ckpts:
        start = ckpts[-1]; agent.load(os.path.join(rd, f"ckpt_{start:06d}.pt"))
        print(f"[{rd}] resuming at step {start} (NOTE: replay buffer is not restored)")
    t0 = time.time(); ep_seed = seed
    for step in range(start + 1, STEPS + 1):
        if step <= WARMUP: a = env.action_space.sample()
        else:              a = agent.act(obs, deterministic=False)
        obs2, r, term, trunc, info = env.step(a)
        done = term or trunc
        agent.observe(obs, a, float(r), np.asarray(info["reward_vec"], dtype=float), obs2, done)
        if step > WARMUP: agent.update(step)
        obs = obs2
        if done:
            ep_seed += 1; obs = reset_env(env, ep_seed, w)
        if step % CKPT_EVERY == 0:
            agent.save(os.path.join(rd, f"ckpt_{step:06d}.pt"))
            print(f"[{rd}] step {step}/{STEPS}  {time.time()-t0:.0f}s", flush=True)

    # ---- 2. validate every checkpoint (VAL seeds only) ----
    ckpts = sorted(int(f[5:11]) for f in os.listdir(rd) if f.startswith("ckpt_"))
    val_rows = []
    for c in ckpts:
        agent.load(os.path.join(rd, f"ckpt_{c:06d}.pt"))
        agg, _ = evaluate(agent, w, VAL_SEEDS)
        val_rows.append(dict(ckpt=c, **agg))
    write_csv(os.path.join(rd, "validation.csv"), val_rows)

    # ---- 3. select highest-scoring SAFE checkpoint, freeze ----
    safe = [r for r in val_rows if r["safe_all"]]
    pool = safe if safe else val_rows            # if none safe, take best & flag it
    best = max(pool, key=lambda r: r["score"])
    sel = dict(algo=algo, pref_i=pref_i, preference=w, seed=seed, ckpt=best["ckpt"],
               val_score=best["score"], any_safe=bool(safe), n_ckpts=len(ckpts),
               val_seeds=VAL_SEEDS, test_seeds=TEST_SEEDS, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(sel, open(os.path.join(rd, "selection.json"), "w"), indent=2)

    # ---- 4. evaluate frozen pick ONCE on TEST seeds ----
    agent.load(os.path.join(rd, f"ckpt_{best['ckpt']:06d}.pt"))
    tagg, trows = evaluate(agent, w, TEST_SEEDS)
    write_csv(os.path.join(rd, "test.csv"), trows)
    res = dict(algo=algo, pref_i=pref_i, seed=seed, ckpt=best["ckpt"], any_safe=bool(safe),
               val_score=best["score"], test_score=tagg["score"],
               selection_bias=best["score"] - tagg["score"],
               test_score_std=tagg["score_std"], safe_frac_test=tagg["safe_frac"],
               Cc=tagg["Cc"], Gp=tagg["Gp"], T12=tagg["T12"], T_std=tagg["T_std"],
               T_max=tagg["T_max"], extraction=tagg["extraction"])
    json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=2)
    open(os.path.join(rd, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    return f"done {rd} ckpt={best['ckpt']} val={best['score']:.2f} test={tagg['score']:.2f}"


# ---------------------------------------------------------------- aggregation
def aggregate():
    rows = []
    for algo in ALGOS:
        for pi in range(len(PREFERENCE_SET)):
            for s in TRAIN_SEEDS:
                p = os.path.join(OUT_DIR, algo, f"pref{pi}", f"seed{s}", "result.json")
                if os.path.exists(p): rows.append(json.load(open(p)))
    if not rows: print("no finished runs yet"); return
    write_csv(os.path.join(OUT_DIR, "results.csv"), rows)
    summ, bias = [], []
    for algo in ALGOS:
        for pi in range(len(PREFERENCE_SET)):
            g = [r for r in rows if r["algo"] == algo and r["pref_i"] == pi]
            if not g: continue
            def ms(k): v = np.array([r[k] for r in g]); return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            d = dict(algo=algo, pref_i=pi, preference=str(PREFERENCE_SET[pi]), n_seeds=len(g))
            for k in ["test_score", "Cc", "Gp", "T12", "T_std", "T_max", "extraction"]:
                m, sd = ms(k); d[f"{k}_mean"] = round(m, 4); d[f"{k}_std"] = round(sd, 4)
            summ.append(d)
            vm, vs = ms("val_score"); tm, ts = ms("test_score"); bm, bs = ms("selection_bias")
            bias.append(dict(algo=algo, pref_i=pi, preference=str(PREFERENCE_SET[pi]), n_seeds=len(g),
                             val_mean=round(vm, 3), test_mean=round(tm, 3),
                             bias_mean=round(bm, 3), bias_std=round(bs, 3)))
    write_csv(os.path.join(OUT_DIR, "summary.csv"), summ)
    write_csv(os.path.join(OUT_DIR, "selection_bias.csv"), bias)
    print(f"\n{len(rows)} runs aggregated -> results.csv, summary.csv, selection_bias.csv")
    print(f"{'algo':<5}{'pref':<6}{'n':<3}{'test_score':>18}{'sel.bias':>12}{'T_max':>8}")
    for d, b in zip(summ, bias):
        print(f"{d['algo']:<5}{d['pref_i']:<6}{d['n_seeds']:<3}"
              f"{d['test_score_mean']:>10.2f}±{d['test_score_std']:<7.2f}{b['bias_mean']:>10.3f}  {d['T_max_mean']:>6.2f}")


# ---------------------------------------------------------------- mock (self-test)
def install_mock():
    """Tiny fake env + agent so the whole pipeline can be verified in seconds."""
    global make_env, make_agent, reset_env, STEPS, WARMUP, CKPT_EVERY, VAL_SEEDS, TEST_SEEDS
    STEPS, WARMUP, CKPT_EVERY = 300, 20, 100
    VAL_SEEDS, TEST_SEEDS = [100, 101, 102], [200, 201, 202, 203]
    class Box:
        def __init__(s, n): s.shape = (n,); s.rng = np.random.default_rng(0)
        def sample(s): return s.rng.uniform(-1, 1, s.shape)
    class MockEnv:
        def __init__(s, seed): s.rng = np.random.default_rng(seed); s.action_space = Box(3); s.t = 0
        def reset(s, seed=None, options=None):
            s.rng = np.random.default_rng(seed); s.t = 0; s.w = np.array(options["preference"])
            return s.rng.normal(size=43).astype(np.float32), {}
        def step(s, a):
            s.t += 1; T = 74 + 2 * float(a[0]) + s.rng.normal(0, 0.1)
            rv = np.array([0.5 + 0.1 * a[0], 0.5 - 0.1 * a[0], 0.6, -max(0, T - T_HI) - max(0, T_LO - T)])
            info = dict(reward_vec=rv, T12=T, Cc=12.0, Gp=1.2 + 0.3 * a[0], extraction=0.97)
            return s.rng.normal(size=43).astype(np.float32), float(rv.sum()), False, s.t >= 30, info
    class MockAgent:
        def __init__(s, seed): s.rng = np.random.default_rng(seed); s.bias = 0.0
        def act(s, obs, deterministic=False): return np.clip(np.array([s.bias, 0, 0]) + (0 if deterministic else s.rng.normal(0, .2, 3)), -1, 1)
        def observe(s, *a): pass
        def update(s, step): s.bias = min(0.8, s.bias + 0.005)   # "learns" to run hotter
        def save(s, p): json.dump(dict(bias=s.bias), open(p, "w"))
        def load(s, p): s.bias = json.load(open(p))["bias"]
    make_env = lambda seed, w: MockEnv(seed)
    make_agent = lambda algo, od, ad, seed, preference=None: MockAgent(seed)
    reset_env = lambda env, seed, w: env.reset(seed=seed, options={"preference": w})[0]


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--prefs", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="continue unfinished runs from last checkpoint (buffer not restored)")
    ap.add_argument("--mock", action="store_true", help="30-second self-test with fake env/agent")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if args.mock: OUT_DIR = "runs_mock"; install_mock()
    if args.aggregate_only: aggregate(); sys.exit(0)

    algos = args.algos or ALGOS
    prefs = args.prefs if args.prefs is not None else list(range(len(PREFERENCE_SET)))
    seeds = args.seeds or TRAIN_SEEDS
    jobs = [(a, p, s, args.resume) for a, p, s in itertools.product(algos, prefs, seeds)]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{len(jobs)} runs | {STEPS} steps each | val seeds {VAL_SEEDS[0]}..{VAL_SEEDS[-1]} "
          f"| test seeds {TEST_SEEDS[0]}..{TEST_SEEDS[-1]} | workers={args.workers}")
    if args.workers > 1 and not args.mock:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for msg in pool.imap_unordered(run_one, jobs): print(msg, flush=True)
    else:
        for j in jobs: print(run_one(j), flush=True)
    aggregate()
