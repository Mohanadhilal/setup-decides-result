#!/usr/bin/env python3
"""
compare_rl_algorithms.py
========================
Compare deep-RL algorithms on the sugar extraction control problem to justify
the choice of SAC. All algorithms are trained under IDENTICAL conditions - the
same number of steps, warmup, batch size, learning rate, network size, discount,
and target-update rate as the SAC fixed-weight policies - at the same balanced
preference [0.34, 0.33, 0.33] (scalarized), for a strictly fair head-to-head.

Algorithms (the standard continuous-control baselines):
  - SAC  (Soft Actor-Critic)     - off-policy, entropy-regularized stochastic
  - TD3  (Twin Delayed DDPG)     - off-policy, deterministic, twin critics
  - PPO  (Proximal Policy Opt.)  - on-policy, stochastic, clipped objective

Produces:
  - runs/rl_comparison.csv        (final metrics per algorithm)
  - runs/rl_learning_curves.png   (reward vs steps)
  - runs/rl_comparison_bars.png   (final performance bars)

Run in PyCharm:
  Run 'compare_rl_algorithms'
Note: trains 3 agents at 150k steps each; on CPU this takes several hours.
Reduce STEPS at the top of the file for a quick functional test.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sugar_extraction_env import SugarExtractionEnv

DEVICE = torch.device("cpu")
# Balanced preference (three-way objectives: extraction, energy, concentration),
# matching the balanced fixed-weight policy used elsewhere.
PREF = [0.34, 0.33, 0.33]
# Training conditions IDENTICAL to the SAC fixed-weight policies, for a fair
# head-to-head comparison (see CONFIG in train_morl_sac.py).
STEPS = 150_000                 # same as SAC fixed-weight training
WARMUP = 5_000                  # same warmup
EVAL_EVERY = 10_000             # same eval cadence
BATCH = 256
GAMMA = 0.99
TAU = 0.005
LR = 2e-4
HIDDEN = 256
SEED = 0


# ============================================================
# Shared networks
# ============================================================
class MLP(nn.Module):
    def __init__(self, sizes, act=nn.ReLU, out_act=nn.Identity):
        super().__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers += [nn.Linear(sizes[i], sizes[i+1]),
                       act() if i < len(sizes)-2 else out_act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.no = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.max, self.ptr, self.n = size, 0, 0

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i, 0], self.no[i], self.d[i, 0] = o, a, r, no, d
        self.ptr = (self.ptr+1) % self.max
        self.n = min(self.n+1, self.max)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, batch)
        t = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return t(self.o), t(self.a), t(self.r), t(self.no), t(self.d)


def scalarize(reward_vec, pref):
    """Scalarize the reward exactly as the environment does.

    The vector now has four channels. The first three are the competing
    objectives and are weighted by the preference; the fourth carries the
    thermal constraint and is added with weight one, outside the
    scalarization, so that a small preference weight cannot dilute it. Taking
    a plain dot product with a three-element preference, as this function did
    before, scores something no agent is optimising.
    """
    v = np.asarray(reward_vec, dtype=float)
    w = np.asarray(pref, dtype=float)
    s = float(np.dot(v[:len(w)], w))
    if len(v) > len(w):
        s += float(np.sum(v[len(w):]))
    return s


# ============================================================
# Environment helper (fixed preference)
# ============================================================
def make_env(seed, domain_randomize=True):
    env = SugarExtractionEnv(seed=seed, enable_disturbance=True,
                             domain_randomize=domain_randomize)
    return env


def reset_env(env, seed=None):
    return env.reset(seed=seed, options={"preference": PREF})


# ============================================================
# SAC
# ============================================================
class SACActor(nn.Module):
    def __init__(self, o, a, h=256):
        super().__init__()
        self.body = MLP([o, h, h])
        self.mu = nn.Linear(h, a)
        self.log_std = nn.Linear(h, a)

    def forward(self, obs):
        x = self.body(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        a = torch.tanh(x)
        logp = normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)


class SAC:
    name = "SAC"
    def __init__(self, o, a, h=256, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma, self.tau = gamma, tau
        self.actor = SACActor(o, a, h).to(DEVICE)
        self.q1 = MLP([o+a, h, h, 1]).to(DEVICE)
        self.q2 = MLP([o+a, h, h, 1]).to(DEVICE)
        self.q1t = MLP([o+a, h, h, 1]).to(DEVICE); self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t = MLP([o+a, h, h, 1]).to(DEVICE); self.q2t.load_state_dict(self.q2.state_dict())
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.qo = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=lr)
        self.target_ent = -float(a)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_o = torch.optim.Adam([self.log_alpha], lr=lr)

    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(o); a = torch.tanh(mu)
            else:
                a, _ = self.actor.sample(o)
        return a.cpu().numpy()[0]

    def update(self, buf, batch=BATCH):
        o, a, r, no, d = buf.sample(batch)
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            na, nlogp = self.actor.sample(no)
            q1t = self.q1t(torch.cat([no, na], -1))
            q2t = self.q2t(torch.cat([no, na], -1))
            qt = torch.min(q1t, q2t) - alpha*nlogp
            target = r + self.gamma*(1-d)*qt
        q1 = self.q1(torch.cat([o, a], -1)); q2 = self.q2(torch.cat([o, a], -1))
        qloss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.qo.zero_grad(); qloss.backward(); self.qo.step()

        pi, logp = self.actor.sample(o)
        q1pi = self.q1(torch.cat([o, pi], -1)); q2pi = self.q2(torch.cat([o, pi], -1))
        aloss = (alpha*logp - torch.min(q1pi, q2pi)).mean()
        self.ao.zero_grad(); aloss.backward(); self.ao.step()

        aloss2 = -(self.log_alpha*(logp+self.target_ent).detach()).mean()
        self.alpha_o.zero_grad(); aloss2.backward(); self.alpha_o.step()

        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1t.parameters()):
                tp.data.mul_(1-self.tau); tp.data.add_(self.tau*p.data)
            for p, tp in zip(self.q2.parameters(), self.q2t.parameters()):
                tp.data.mul_(1-self.tau); tp.data.add_(self.tau*p.data)


# ============================================================
# DDPG / TD3 (deterministic)
# ============================================================
class DetActor(nn.Module):
    def __init__(self, o, a, h=256):
        super().__init__()
        self.net = MLP([o, h, h, a])

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class TD3:
    name = "TD3"
    def __init__(self, o, a, h=256, lr=3e-4, gamma=0.99, tau=0.005,
                 noise=0.1, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.gamma, self.tau, self.noise, self.act_dim = gamma, tau, noise, a
        self.policy_noise, self.noise_clip, self.policy_delay = policy_noise, noise_clip, policy_delay
        self.actor = DetActor(o, a, h).to(DEVICE)
        self.actor_t = DetActor(o, a, h).to(DEVICE); self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = MLP([o+a, h, h, 1]).to(DEVICE)
        self.q2 = MLP([o+a, h, h, 1]).to(DEVICE)
        self.q1t = MLP([o+a, h, h, 1]).to(DEVICE); self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t = MLP([o+a, h, h, 1]).to(DEVICE); self.q2t.load_state_dict(self.q2.state_dict())
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.qo = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=lr)
        self.it = 0

    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(o).cpu().numpy()[0]
        if not deterministic:
            a = np.clip(a + self.noise*np.random.randn(self.act_dim), -1, 1)
        return a

    def update(self, buf, batch=BATCH):
        self.it += 1
        o, a, r, no, d = buf.sample(batch)
        with torch.no_grad():
            noise = (torch.randn_like(a)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_t(no)+noise).clamp(-1, 1)
            q1t = self.q1t(torch.cat([no, na], -1))
            q2t = self.q2t(torch.cat([no, na], -1))
            target = r + self.gamma*(1-d)*torch.min(q1t, q2t)
        q1 = self.q1(torch.cat([o, a], -1)); q2 = self.q2(torch.cat([o, a], -1))
        qloss = F.mse_loss(q1, target)+F.mse_loss(q2, target)
        self.qo.zero_grad(); qloss.backward(); self.qo.step()

        # Delayed policy update
        if self.it % self.policy_delay == 0:
            aloss = -self.q1(torch.cat([o, self.actor(o)], -1)).mean()
            self.ao.zero_grad(); aloss.backward(); self.ao.step()
            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_t.parameters()):
                    tp.data.mul_(1-self.tau); tp.data.add_(self.tau*p.data)
                for p, tp in zip(self.q1.parameters(), self.q1t.parameters()):
                    tp.data.mul_(1-self.tau); tp.data.add_(self.tau*p.data)
                for p, tp in zip(self.q2.parameters(), self.q2t.parameters()):
                    tp.data.mul_(1-self.tau); tp.data.add_(self.tau*p.data)


# ============================================================
# PPO (on-policy)
# ============================================================
class PPOActorCritic(nn.Module):
    def __init__(self, o, a, h=256):
        super().__init__()
        self.body = MLP([o, h, h])
        self.mu = nn.Linear(h, a)
        self.log_std = nn.Parameter(torch.zeros(a))
        self.v = MLP([o, h, h, 1])

    def forward(self, obs):
        x = self.body(obs)
        return self.mu(x), self.log_std.exp(), self.v(obs)

    def act(self, obs):
        mu, std, v = self(obs)
        dist = torch.distributions.Normal(mu, std)
        x = dist.sample()
        a = torch.tanh(x)
        logp = (dist.log_prob(x) - torch.log(1-a.pow(2)+1e-6)).sum(-1)
        return a, logp, v.squeeze(-1)

    def evaluate(self, obs, act):
        mu, std, v = self(obs)
        # invert tanh
        x = torch.atanh(torch.clamp(act, -0.999, 0.999))
        dist = torch.distributions.Normal(mu, std)
        logp = (dist.log_prob(x) - torch.log(1-act.pow(2)+1e-6)).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, v.squeeze(-1), ent


class PPO:
    name = "PPO"
    def __init__(self, o, a, h=256, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, rollout=2048):
        self.gamma, self.lam, self.clip, self.epochs, self.rollout = gamma, lam, clip, epochs, rollout
        self.ac = PPOActorCritic(o, a, h).to(DEVICE)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _, _ = self.ac(o); a = torch.tanh(mu)
                return a.cpu().numpy()[0]
            a, logp, v = self.ac.act(o)
        self._last = (logp.item(), v.item())
        return a.cpu().numpy()[0]

    def update_from_rollout(self, obs, acts, rews, dones, vals, logps):
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=DEVICE)
        acts = torch.as_tensor(np.array(acts), dtype=torch.float32, device=DEVICE)
        logps = torch.as_tensor(np.array(logps), dtype=torch.float32, device=DEVICE)
        # GAE
        adv = np.zeros(len(rews), np.float32)
        last = 0.0
        for t in reversed(range(len(rews))):
            nextv = vals[t+1] if t+1 < len(vals) else 0.0
            delta = rews[t] + self.gamma*nextv*(1-dones[t]) - vals[t]
            last = delta + self.gamma*self.lam*(1-dones[t])*last
            adv[t] = last
        ret = adv + np.array(vals[:len(rews)], np.float32)
        adv = (adv - adv.mean())/(adv.std()+1e-8)
        adv = torch.as_tensor(adv, device=DEVICE)
        ret = torch.as_tensor(ret, device=DEVICE)

        for _ in range(self.epochs):
            logp, v, ent = self.ac.evaluate(obs, acts)
            ratio = (logp - logps).exp()
            s1 = ratio*adv
            s2 = torch.clamp(ratio, 1-self.clip, 1+self.clip)*adv
            ploss = -torch.min(s1, s2).mean()
            vloss = F.mse_loss(v, ret)
            loss = ploss + 0.5*vloss - 0.01*ent.mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()


# ============================================================
# Training loops
# ============================================================
def evaluate_agent(agent, eval_env, episodes=3):
    Cc, Gp, T12, extr, Tstd, R = [], [], [], [], [], []
    for _ in range(episodes):
        obs, _ = reset_env(eval_env)
        Th = []; last = None; total = 0.0; done = False
        while not done:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(a)
            total += scalarize(info["reward_vec"], PREF)
            Th.append(info["T12"]); last = info
            done = term or trunc
        Cc.append(last["Cc"]); Gp.append(last["Gp"]); T12.append(last["T12"])
        extr.append(last["extraction"]); Tstd.append(np.std(Th[-50:])); R.append(total)
    return dict(Cc=np.mean(Cc), Gp=np.mean(Gp), T12=np.mean(T12),
                extraction=np.mean(extr), T_std=np.mean(Tstd), reward=np.mean(R))


def train_offpolicy(agent, steps=STEPS, warmup=WARMUP):
    """Train an off-policy agent (SAC/TD3/DDPG)."""
    env = make_env(SEED)
    eval_env = make_env(SEED+100, domain_randomize=False)
    o = env.observation_space.shape[0]; a = env.action_space.shape[0]
    buf = ReplayBuffer(300_000, o, a)
    obs, _ = reset_env(env, SEED)
    history = []
    t0 = time.time()
    for step in range(1, steps+1):
        if step < warmup:
            act = env.action_space.sample()
        else:
            act = agent.act(obs)
        no, _, term, trunc, info = env.step(act)
        r = scalarize(info["reward_vec"], PREF)
        buf.add(obs, act, r, no, float(term))
        obs = no
        if term or trunc:
            obs, _ = reset_env(env)
        if step >= warmup:
            agent.update(buf)
        if step % EVAL_EVERY == 0:
            m = evaluate_agent(agent, eval_env)
            m["step"] = step; history.append(m)
            print(f"  [{agent.name}] step={step:>6} reward={m['reward']:.1f} "
                  f"Cc={m['Cc']:.2f} Gp={m['Gp']:.3f} T12={m['T12']:.1f} "
                  f"Tstd={m['T_std']:.3f} | {time.time()-t0:.0f}s")
    return history


def train_ppo(agent, steps=STEPS):
    """Train PPO (on-policy)."""
    env = make_env(SEED)
    eval_env = make_env(SEED+100, domain_randomize=False)
    obs, _ = reset_env(env, SEED)
    history = []
    t0 = time.time()
    step = 0
    while step < steps:
        obs_b, act_b, rew_b, done_b, val_b, logp_b = [], [], [], [], [], []
        for _ in range(agent.rollout):
            act = agent.act(obs)
            logp, v = agent._last
            no, _, term, trunc, info = env.step(act)
            r = scalarize(info["reward_vec"], PREF)
            obs_b.append(obs); act_b.append(act); rew_b.append(r)
            done_b.append(float(term)); val_b.append(v); logp_b.append(logp)
            obs = no; step += 1
            if term or trunc:
                obs, _ = reset_env(env)
        agent.update_from_rollout(obs_b, act_b, rew_b, done_b, val_b, logp_b)
        if step // EVAL_EVERY > len(history):
            m = evaluate_agent(agent, eval_env)
            m["step"] = step; history.append(m)
            print(f"  [PPO] step={step:>6} reward={m['reward']:.1f} "
                  f"Cc={m['Cc']:.2f} Gp={m['Gp']:.3f} T12={m['T12']:.1f} "
                  f"Tstd={m['T_std']:.3f} | {time.time()-t0:.0f}s")
    return history


# ============================================================
def main():
    os.makedirs("runs", exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    env = make_env(SEED)
    o = env.observation_space.shape[0]; a = env.action_space.shape[0]

    print("="*64)
    print("RL Algorithm Comparison on Sugar Extraction (balanced pref)")
    print(f"Steps per algorithm: {STEPS}, preference: {PREF}")
    print("="*64)

    algos = {
        "SAC": (SAC(o, a, h=HIDDEN, lr=LR, gamma=GAMMA, tau=TAU), "offpolicy"),
        "TD3": (TD3(o, a, h=HIDDEN, lr=LR, gamma=GAMMA, tau=TAU), "offpolicy"),
        "PPO": (PPO(o, a, h=HIDDEN, lr=LR, gamma=GAMMA), "ppo"),
    }

    histories = {}
    finals = {}
    for name, (agent, kind) in algos.items():
        print(f"\n--- Training {name} ---")
        np.random.seed(SEED); torch.manual_seed(SEED)
        if kind == "ppo":
            h = train_ppo(agent)
        else:
            h = train_offpolicy(agent)
        histories[name] = h
        finals[name] = h[-1] if h else {}

    # Save CSV
    with open("runs/rl_comparison.csv", "w") as f:
        f.write("Algorithm,reward,Cc,Gp,T12,extraction,T_std\n")
        for name, m in finals.items():
            f.write(f"{name},{m.get('reward',0):.4f},{m.get('Cc',0):.4f},"
                    f"{m.get('Gp',0):.4f},{m.get('T12',0):.4f},"
                    f"{m.get('extraction',0):.4f},{m.get('T_std',0):.4f}\n")
    print("\nSaved: runs/rl_comparison.csv")

    # Print final table
    print("\n" + "="*72)
    print(f"{'Algorithm':<10}{'Reward':>9}{'Cc,%':>8}{'Gp':>8}{'T12':>8}"
          f"{'Tstd':>8}{'Extr,%':>8}")
    print("-"*72)
    for name, m in finals.items():
        print(f"{name:<10}{m.get('reward',0):>9.1f}{m.get('Cc',0):>8.2f}"
              f"{m.get('Gp',0):>8.3f}{m.get('T12',0):>8.1f}"
              f"{m.get('T_std',0):>8.3f}{m.get('extraction',0)*100:>8.1f}")
    print("="*72)

    # Learning curves
    plt.figure(figsize=(10, 6))
    for name, h in histories.items():
        if h:
            steps = [m["step"] for m in h]
            rewards = [m["reward"] for m in h]
            plt.plot(steps, rewards, "o-", label=name, lw=2, ms=5)
    plt.xlabel("Training steps"); plt.ylabel("Evaluation reward (scalarized)")
    plt.title("RL Algorithm Learning Curves (Sugar Extraction, balanced pref)")
    plt.legend(fontsize=11); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/rl_learning_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: runs/rl_learning_curves.png")
    plt.close()

    # Final performance bars
    names = list(finals.keys())
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    def bars(a_, key, title, ylabel, scale=1.0):
        vals = [finals[n].get(key, 0)*scale for n in names]
        a_.bar(names, vals, color=colors, edgecolor="black", alpha=0.85)
        a_.set_title(title); a_.set_ylabel(ylabel); a_.grid(axis="y", alpha=0.3)
    bars(ax[0,0], "reward", "Final Reward (higher=better)", "reward")
    bars(ax[0,1], "Gp", "Steam Consumption (lower=better)", "Gp, kg/s")
    bars(ax[1,0], "T_std", "Temperature Stability (lower=better)", "std(T12), degC")
    bars(ax[1,1], "Cc", "Sucrose Concentration", "Cc, %")
    plt.suptitle("RL Algorithm Comparison: Final Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("runs/rl_comparison_bars.png", dpi=150, bbox_inches="tight")
    print("Saved: runs/rl_comparison_bars.png")
    plt.close()


if __name__ == "__main__":
    main()
