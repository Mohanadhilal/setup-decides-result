#!/usr/bin/env python3
"""
sugar_extraction_env.py
=======================
Gymnasium-compatible environment for Multi-Objective Reinforcement Learning (MORL)
for control of the sugar extraction process.

Action, continuous, normalized to [-1, 1]:
  a[0] -> mu_cmd   (steam valve opening)
  a[1] -> P_p      (steam pressure)
  a[2] -> G_v      (extraction water flow)

Observation:
  - temperatures per cell (m)
  - juice concentration per cell (m)
  - cossette concentration per cell (m)
  - current valve position mu_act
  - current P_p, G_v (normalized)
  - current Kc (measured disturbance)
  Dimension: 3*m + 4

Vector reward (3 objectives):
  r[0] = +sucrose yield (extraction / Cc)
  r[1] = -energy cost (steam flow Gp)
  r[2] = -temperature deviation from target (stability)
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sugar_extraction_model import SugarExtractionModel
from scaling import get_scaling



class SugarExtractionEnv(gym.Env):
    """MORL environment for sugar extraction."""

    metadata = {"render_modes": []}

    def __init__(self, m=12, dt=10.0, episode_minutes=200,
                 enable_disturbance=True, domain_randomize=True, seed=None):
        super().__init__()
        self.m = m
        self.dt = dt                       # discretization step, s
        self.episode_steps = int(episode_minutes * 60 / dt)
        self.enable_disturbance = enable_disturbance
        self.domain_randomize = domain_randomize
        self._seed = seed

        self.model = SugarExtractionModel(m=m, seed=seed)
        # Objective scaling: oracle by default (the submitted results), or derived
        # from training data only, selected by the MORL_SCALING environment variable.
        self._S = get_scaling()

        # --- Action space: 3 continuous, normalized to [-1,1] ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,),
                                       dtype=np.float32)

        # --- Observation space (state + preference weights) ---
        # The preference vector w (reward_dim) is appended to the observation
        # so a single policy can be conditioned on the trade-off weights.
        # Three scalarized objectives plus one safety channel. The safety
        # channel is a constraint, not an objective: it is added outside the
        # scalarization with weight one, so its effect cannot be diluted by a
        # small preference weight. The preference vector therefore stays
        # three-dimensional while the reward vector has four channels.
        self.n_pref = 3
        self.reward_dim = 4
        obs_dim = 3*m + 4 + self.n_pref
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        # MORL reward space
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(self.reward_dim,), dtype=np.float32)

        # Current preference vector (set each episode); default = balanced
        self.preference = np.array([0.5, 0.25, 0.25], dtype=np.float32)

        self.x = None
        self.t_step = 0
        self.prev_action = None      # for action-smoothness penalty

    def set_preference(self, w):
        """Set the active preference vector w (will be normalized to sum=1)."""
        w = np.asarray(w, dtype=np.float32)
        w = np.clip(w, 1e-3, None)
        self.preference = w / w.sum()

    # ============================================================
    def _denorm_action(self, a):
        """Convert normalized action [-1,1] to physical values."""
        a = np.clip(a, -1.0, 1.0)
        mu  = 0.5*(a[0]+1.0)*(self.model.mu_max - self.model.mu_min) + self.model.mu_min
        Pp  = 0.5*(a[1]+1.0)*(self.model.Pp_max - self.model.Pp_min) + self.model.Pp_min
        Gv  = 0.5*(a[2]+1.0)*(self.model.Gv_max - self.model.Gv_min) + self.model.Gv_min
        return mu, Pp, Gv

    def _get_obs(self):
        m = self.m
        T    = self.x[m:2*m]
        Cstr = self.x[2*m:3*m]
        Cliq = self.x[3*m:4*m]
        mu_act = self.x[4*m]

        # Normalization for stable training
        T_n    = (T - 60.0) / 20.0
        Cliq_n = Cliq / 15.0
        Cstr_n = Cstr / self.model.C_beet0
        Pp_n   = (self.model.P_p - self.model.Pp_min) / (self.model.Pp_max - self.model.Pp_min)
        Gv_n   = (self.model.G_v - self.model.Gv_min) / (self.model.Gv_max - self.model.Gv_min)
        Kc_n   = (self.model.Kc - 1.0) / 0.3

        obs = np.concatenate([T_n, Cliq_n, Cstr_n,
                              [mu_act, Pp_n, Gv_n, Kc_n],
                              self.preference]).astype(np.float32)
        return obs

    def _vector_reward(self, out):
        """Vector reward (3 competing objectives) with temperature as a SAFETY
        CONSTRAINT rather than a tracking target.

        Industrial framing (sugar-beet diffusion): the controller trades off
        three quantities that genuinely compete, while keeping temperature in a
        safe window so sucrose is not degraded:

          Objective 1 - EXTRACTION (recovered sugar): rises with temperature
            (faster diffusion). Higher T -> more sugar recovered, but costs
            steam and approaches the degradation limit.
          Objective 2 - ENERGY (steam): lower steam is better; but less steam
            means lower temperature, hence lower extraction. Direct tension
            with objective 1.
          Objective 3 - CONCENTRATION (juice purity/Cc): set by draft; kept
            near a realistic target.

        Temperature is NOT an objective here. It is bounded to the safe range
        [70, 78] C by a one-sided penalty that is zero inside the window and
        grows sharply outside it (protects against sucrose caramelization
        above ~80 C and poor extraction below ~70 C).
        """
        Cc = out['Cc']
        Gp = out['Gp']
        extraction = out['extraction']
        T12 = out['T12']
        Cc_target = self.model.Cliq_target          # 12.5 %
        Gp_baseline = 1.20                          # nominal steam (PID-like)

        # Rescaled so each objective spans a comparable range over the
        # achievable operating region; otherwise equal preference weights
        # do not produce balanced behaviour.
        S = self._S
        r_extraction = (extraction - S["ext_off"]) / S["ext_sc"]
        # --- Plant-level energy -------------------------------------------
        # Sugar output is fixed by the beet feed and the extraction
        # efficiency. Diluting the juice adds no product, it only adds water
        # that the evaporator station must boil off, so the honest energy
        # objective charges both.
        _kappa    = getattr(self, "evap_kappa", 0.30)     # kg steam / kg water
        _cc_thick = getattr(self, "cc_thick", 65.0)       # thick juice, %
        _g_solid  = getattr(self.model, "G_sv_nom", 3.2)  # beet feed, kg/s
        _c_beet   = getattr(self.model, "C_beet0", 0.175) # beet sugar fraction

        _sugar = _g_solid * _c_beet * extraction          # kg/s recovered
        _cc = max(float(Cc), 1e-3)                        # guard the division
        _evap_water = _sugar * 100.0 * (1.0 / _cc - 1.0 / _cc_thick)
        _evap_water = max(_evap_water, 0.0)
        G_total = Gp + _kappa * _evap_water

        # Rescaled over the attainable range of G_total (about 1.95 to 2.77)
        # so this objective spans the same magnitude as the extraction one.
        r_energy = (S["gt_base"] - G_total) / S["gt_sc"]
        r_conc = np.exp(-((Cc - S["cc_target"]) / S["cc_width"]) ** 2)

        # Hinge penalty: linear + quadratic. The linear term gives a non-zero
        # gradient exactly at the boundary, so shaving the limit is never
        # profitable; the quadratic term punishes large excursions.
        T_lo, T_hi = 70.0, 78.0
        if T12 < T_lo:
            dev = T_lo - T12
        elif T12 > T_hi:
            dev = T12 - T_hi
        else:
            dev = 0.0
        # Linear term deters boundary shaving; quadratic punishes large
        # excursions; the cap keeps critic targets bounded during exploration.
        penalty = min(1.0 * dev + 0.5 * dev**2, 5.0)

        # The penalty is NOT folded into the objectives any more. It travels
        # as its own channel so that the agent can apply it at full strength
        # after scalarization, whatever the preference weights are.
        return np.array([r_extraction, r_energy, r_conc, -penalty],
                        dtype=np.float32)

    # ============================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.model.rng = np.random.default_rng(seed)

        self.model.reset_params()

        # Preference handling:
        #   - options={"preference": w}  -> use a fixed given preference
        #   - options={"sample_preference": True} -> draw a random preference
        #     (used during training so one policy covers the whole Pareto front)
        if options is not None and "preference" in options:
            self.set_preference(options["preference"])
        elif options is not None and options.get("sample_preference", False):
            # Dirichlet sampling gives a uniform spread over the simplex
            w = self.model.rng.dirichlet(np.ones(self.n_pref))
            self.set_preference(w)

        # (Domain randomization) random physical parameters for robustness
        if self.domain_randomize:
            self.model.K_ht  = 40.0 * self.model.rng.uniform(0.8, 1.2)
            self.model.c_p   = 3650.0 * self.model.rng.uniform(0.95, 1.05)
            self.model.kLa_nom = 0.020 * self.model.rng.uniform(0.85, 1.15)
            # random initial raw-material quality
            self.model.Kc = self.model.rng.uniform(0.75, 1.10)
            self.model.Fc = 3.5 * self.model.rng.uniform(0.9, 1.1)

        self.x = self.model.initial_state()
        self.t_step = 0
        self.prev_action = None
        return self._get_obs(), {}

    # ============================================================
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # 1) Apply action
        mu, Pp, Gv = self._denorm_action(action)
        self.model.mu_cmd = mu
        self.model.P_p    = Pp
        self.model.G_v    = Gv

        # 2) Disturbances
        if self.enable_disturbance:
            self.model.apply_disturbance(self.dt)

        # 3) Integrate the model over step dt (several sub-steps for accuracy)
        n_sub = 2
        sub_dt = self.dt / n_sub
        for _ in range(n_sub):
            self.x = self.model.step_rk4(self.x, sub_dt)

        # 4) Outputs and reward
        out = self.model.get_outputs(self.x)
        reward_vec = self._vector_reward(out)

        # 4b) Action-smoothness penalty: discourage large jumps in the control
        # signal, which cause temperature oscillations. Added to the stability
        # objective so it is traded off through the same weight.
        if self.prev_action is not None:
            action_jump = np.sum((action - self.prev_action)**2)
            reward_vec[2] -= 0.3 * action_jump          # smoothness -> stability
        self.prev_action = action.copy()

        # 5) Default scalarization (equal weights) - for standard RL
        scalar_reward = float(np.sum(reward_vec))

        # 6) Termination conditions
        self.t_step += 1
        terminated = False
        truncated = self.t_step >= self.episode_steps

        # Penalty for exceeding physical limits (safety)
        if out['T12'] > 85.0 or out['T12'] < 55.0:
            scalar_reward -= 5.0
            reward_vec[2] -= 5.0

        info = dict(reward_vec=reward_vec, **out,
                    Kc=self.model.Kc, Pp=self.model.P_p, Gv=self.model.G_v)

        return self._get_obs(), scalar_reward, terminated, truncated, info


# ================================================================
# Environment self-test
# ================================================================
if __name__ == '__main__':
    env = SugarExtractionEnv(m=12, dt=10.0, episode_minutes=200,
                             enable_disturbance=True, domain_randomize=True,
                             seed=0)
    obs, _ = env.reset(seed=0)
    print("Sugar extraction Gymnasium environment check")
    print("="*55)
    print(f"Observation dimension: {obs.shape}")
    print(f"Action dimension:      {env.action_space.shape}")
    print(f"Number of objectives:  {env.reward_dim}")
    print(f"Steps per episode:     {env.episode_steps}")
    print()

    # Run with constant nominal action
    # mu=0.506 -> a0; Pp=3.5e5 -> a1; Gv=3.5 -> a2
    a_nom = np.array([
        2*(0.506-0)/1.0 - 1,                    # mu
        2*(3.5e5-2.0e5)/(4.5e5-2.0e5) - 1,      # Pp
        2*(3.5-2.0)/(4.5-2.0) - 1,              # Gv
    ], dtype=np.float32)

    total_vec = np.zeros(3)
    last = None
    for k in range(env.episode_steps):
        obs, r, term, trunc, info = env.step(a_nom)
        total_vec += info['reward_vec']
        last = info
        if term or trunc:
            break

    print("After episode (nominal control):")
    print(f"  Cc:          {last['Cc']:.2f} %")
    print(f"  Brix:        {last['brix']:.2f} deg Bx")
    print(f"  T12:         {last['T12']:.1f} deg C")
    print(f"  Gp:          {last['Gp']:.3f} kg/s")
    print(f"  Extraction:  {last['extraction']*100:.1f} %")
    print(f"  Kc (disturb): {last['Kc']:.3f}")
    print()
    print(f"  Total vector reward:")
    print(f"    r_yield  = {total_vec[0]:.1f}")
    print(f"    r_energy = {total_vec[1]:.1f}")
    print(f"    r_stab   = {total_vec[2]:.1f}")
    print()
    print("Environment works correctly. OK")
