#!/usr/bin/env python3
"""
baseline_controllers.py
=======================
Classical / baseline controllers for comparison with MORL-SAC
on the sugar extraction environment.

Implemented controllers:
  1. PIDController            - classical PID on T12 (manipulates mu)
  2. MPCController            - Model Predictive Control (receding horizon)
  3. FuzzyController          - Mamdani-type fuzzy logic controller
  4. TwoLevelHierarchical     - method from the author's previous PhD
                               dissertation (PSA hydrogen): two-level upper/
                               lower model + Nelder-Mead penalty optimization
  5. (MORL-SAC is loaded separately from the trained actor)

All controllers expose the same interface:
    action = controller.act(obs, info)
returning a normalized action in [-1, 1]^3 = (mu, Pp, Gv).
"""

import numpy as np
from scipy.optimize import minimize


# ============================================================
# Action normalization helpers (must match the environment)
# ============================================================
MU_MIN, MU_MAX = 0.0, 1.0
PP_MIN, PP_MAX = 2.0e5, 4.5e5
GV_MIN, GV_MAX = 3.2, 3.8   # must match the narrowed range in the model

def norm_mu(mu):  return 2*(mu - MU_MIN)/(MU_MAX - MU_MIN) - 1
def norm_pp(pp):  return 2*(pp - PP_MIN)/(PP_MAX - PP_MIN) - 1
def norm_gv(gv):  return 2*(gv - GV_MIN)/(GV_MAX - GV_MIN) - 1


# ============================================================
# 1. PID Controller
# ============================================================
class PIDController:
    """Classical PID controller regulating T12 to the target by manipulating
    the steam valve mu. Pp and Gv are held at nominal values."""

    def __init__(self, T_target=73.5, Kp=0.04, Ki=0.002, Kd=0.01,
                 Pp_fixed=3.5e5, Gv_fixed=3.5, dt=10.0):
        self.T_target = T_target
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Pp_fixed = Pp_fixed
        self.Gv_fixed = Gv_fixed
        self.dt = dt
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    def act(self, obs, info):
        T12 = info["T12"]
        err = self.T_target - T12
        self.integral += err * self.dt
        self.integral = np.clip(self.integral, -500, 500)   # anti-windup
        deriv = (err - self.prev_err) / self.dt
        self.prev_err = err

        # PID output -> valve position around a 0.5 bias
        mu = 0.5 + self.Kp*err + self.Ki*self.integral + self.Kd*deriv
        mu = np.clip(mu, MU_MIN, MU_MAX)

        return np.array([norm_mu(mu), norm_pp(self.Pp_fixed),
                         norm_gv(self.Gv_fixed)], dtype=np.float32)


# ============================================================
# 2. MPC Controller (receding horizon)
# ============================================================
class MPCController:
    """Model Predictive Control. Uses a simplified internal first-order model
    of T12 response to mu, optimizes a short control horizon to track the
    temperature target while penalizing steam usage."""

    def __init__(self, model, T_target=73.5, horizon=5, dt=10.0,
                 w_track=1.0, w_energy=0.3, w_move=0.05,
                 Pp_fixed=3.5e5, Gv_fixed=3.5):
        self.model = model           # reference to physical model (for prediction)
        self.T_target = T_target
        self.N = horizon
        self.dt = dt
        self.w_track = w_track
        self.w_energy = w_energy
        self.w_move = w_move
        self.Pp_fixed = Pp_fixed
        self.Gv_fixed = Gv_fixed
        self.prev_mu = 0.506

    def reset(self):
        self.prev_mu = 0.506

    def _predict_T12(self, mu_seq, T12_0):
        """Simple first-order prediction of T12 given a steam valve sequence.
        Identified gain/time-constant from the plant (approx)."""
        K_gain = 18.0      # deg C per unit mu (approx static gain)
        tau = 540.0        # s (cell time constant)
        T_ss_base = 55.0   # baseline temperature at mu=0
        T = T12_0
        traj = []
        for mu in mu_seq:
            T_target_ss = T_ss_base + K_gain * mu
            T = T + (self.dt/tau)*(T_target_ss - T)
            traj.append(T)
        return np.array(traj)

    def _cost(self, mu_seq, T12_0):
        traj = self._predict_T12(mu_seq, T12_0)
        track = np.sum((traj - self.T_target)**2)
        energy = np.sum(np.array(mu_seq)**2)
        move = np.sum(np.diff(np.concatenate([[self.prev_mu], mu_seq]))**2)
        return self.w_track*track + self.w_energy*energy + self.w_move*move

    def act(self, obs, info):
        T12_0 = info["T12"]
        x0 = np.full(self.N, self.prev_mu)
        bounds = [(MU_MIN, MU_MAX)] * self.N
        res = minimize(self._cost, x0, args=(T12_0,),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 30})
        mu_opt = res.x[0]
        self.prev_mu = mu_opt
        return np.array([norm_mu(mu_opt), norm_pp(self.Pp_fixed),
                         norm_gv(self.Gv_fixed)], dtype=np.float32)


# ============================================================
# 3. Fuzzy Logic Controller (Mamdani)
# ============================================================
class FuzzyController:
    """Mamdani-type fuzzy controller. Inputs: temperature error and its rate.
    Output: change in valve position. Uses triangular membership functions
    and centroid defuzzification."""

    def __init__(self, T_target=73.5, Pp_fixed=3.5e5, Gv_fixed=3.5, dt=10.0):
        self.T_target = T_target
        self.Pp_fixed = Pp_fixed
        self.Gv_fixed = Gv_fixed
        self.dt = dt
        self.mu = 0.506
        self.prev_err = 0.0

    def reset(self):
        self.mu = 0.506
        self.prev_err = 0.0

    @staticmethod
    def _tri(x, a, b, c):
        """Triangular membership."""
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a)/(b - a)
        return (c - x)/(c - b)

    def _fuzzify_error(self, e):
        # e in deg C; sets: Negative, Zero, Positive
        return {
            "N": self._tri(e, -10, -5, 0),
            "Z": self._tri(e, -2.5, 0, 2.5),
            "P": self._tri(e, 0, 5, 10),
        }

    def _fuzzify_rate(self, de):
        return {
            "N": self._tri(de, -1.0, -0.5, 0),
            "Z": self._tri(de, -0.25, 0, 0.25),
            "P": self._tri(de, 0, 0.5, 1.0),
        }

    def act(self, obs, info):
        T12 = info["T12"]
        e = self.T_target - T12
        de = (e - self.prev_err) / self.dt
        self.prev_err = e

        fe = self._fuzzify_error(e)
        fd = self._fuzzify_rate(de)

        # Rule base -> output delta-mu singletons:
        # Big increase (+0.08), Small increase (+0.03), Zero (0),
        # Small decrease (-0.03), Big decrease (-0.08)
        rules = [
            (min(fe["P"], fd["P"]),  0.08),
            (min(fe["P"], fd["Z"]),  0.05),
            (min(fe["P"], fd["N"]),  0.02),
            (min(fe["Z"], fd["P"]),  0.02),
            (min(fe["Z"], fd["Z"]),  0.00),
            (min(fe["Z"], fd["N"]), -0.02),
            (min(fe["N"], fd["P"]), -0.02),
            (min(fe["N"], fd["Z"]), -0.05),
            (min(fe["N"], fd["N"]), -0.08),
        ]
        num = sum(w*v for w, v in rules)
        den = sum(w for w, v in rules) + 1e-9
        dmu = num/den

        self.mu = np.clip(self.mu + dmu, MU_MIN, MU_MAX)
        return np.array([norm_mu(self.mu), norm_pp(self.Pp_fixed),
                         norm_gv(self.Gv_fixed)], dtype=np.float32)


# ============================================================
# 4. Two-Level Hierarchical Controller (previous PhD method)
# ============================================================
class TwoLevelHierarchical:
    """
    Reproduction of the control method from the author's previous PhD
    dissertation on PSA hydrogen concentration.

    Architecture:
      - UPPER level: optimizes control variables (mu, Pp, Gv) for nominal
        disturbances using a complex model + penalty-function optimization
        (Nelder-Mead). Compensates low-frequency disturbances.
      - LOWER level: when measured disturbance deviation exceeds a threshold,
        finds corrections (delta) to the optimal control using a simplified
        model. Compensates high-frequency disturbances.

    Objective: maximize a weighted criterion (yield + extraction - energy)
    subject to constraints, via penalty functions (as in the dissertation).
    """

    def __init__(self, model, dt=10.0, threshold=0.03,
                 w_yield=1.0, w_energy=0.3, recompute_every=30):
        self.model = model
        self.dt = dt
        self.threshold = threshold       # |delta_d| threshold (eps)
        self.w_yield = w_yield
        self.w_energy = w_energy
        self.recompute_every = recompute_every
        self.u_opt = np.array([0.506, 3.5e5, 3.5])  # mu, Pp, Gv
        self.Kc_nominal = 1.0
        self.step_count = 0

    def reset(self):
        self.u_opt = np.array([0.506, 3.5e5, 3.5])
        self.Kc_nominal = 1.0
        self.step_count = 0

    def _steady_state_predict(self, u, Kc):
        """Predict steady-state outputs (Cc, T12, Gp) for a given control u
        and disturbance Kc, using a fast surrogate calibrated to the full
        physical model.

        Key physical fact (calibrated from the plant): juice concentration Cc
        saturates around 12 % and is only weakly sensitive to temperature above
        ~72 C, because extraction is already ~97 %. Raising the steam valve
        beyond what is needed for ~73 C does NOT increase Cc - it only wastes
        steam and overheats the apparatus. The surrogate captures this
        saturation so the optimizer does not chase phantom yield gains."""
        mu, Pp, Gv = u
        # Steam flow (eq 4.10)
        Gp = mu * 0.0050 * np.sqrt(max(Pp - 1.1e5, 0))
        # T12 surrogate - coefficients calibrated by least-squares fit to the
        # full physical model (counter-current, m=12 cells).
        T12 = 39.65 + 24.23*mu + 3.148*(Pp - 1.1e5)/1e5 + 3.939*Gv
        T12 = float(np.clip(T12, 40, 95))
        # Cc surrogate: SATURATING in temperature (calibrated). Cc rises with T
        # up to ~72 C then plateaus near 12 %. Also depends on draft (water
        # flow Gv): less water -> higher concentration.
        Cc_max = 12.3 * Kc                              # quality-scaled ceiling
        f_T = 1.0 / (1.0 + np.exp(-(T12 - 64.0)/3.0))   # logistic, ~1 above 70C
        f_draft = (4.5 / (Gv + 1.0))                    # less water -> higher Cc
        Cc = Cc_max * f_T * f_draft
        Cc = float(np.clip(Cc, 0, 14.0))
        return Cc, T12, Gp

    def _objective(self, u, Kc):
        """Penalty-function objective (to MINIMIZE), as in the dissertation
        (eq 4.20-4.21): maximize a weighted criterion subject to constraints
        via penalty functions.

        Corrected criterion: reach the concentration target while keeping
        temperature in the technological window and minimizing steam. Because
        Cc saturates, the optimizer now correctly settles at the minimum steam
        that achieves the target temperature, instead of driving the valve
        fully open."""
        mu, Pp, Gv = u
        Cc, T12, Gp = self._steady_state_predict(u, Kc)

        Cc_target = 12.0
        T_target = 73.5

        # Main criterion (to MINIMIZE): meet Cc target (no benefit beyond it),
        # track temperature, minimize steam.
        yield_term  = -self.w_yield * min(Cc, Cc_target)
        temp_term   = 0.5 * (T12 - T_target)**2
        energy_term = self.w_energy * Gp * 5.0
        # Regularization: prefer nominal steam pressure (avoids degenerate
        # solutions where mu is maxed out with very low Pp at equal steam).
        reg_Pp = 0.5 * ((Pp - 3.5e5)/1e5)**2
        J = yield_term + temp_term + energy_term + reg_Pp

        # Penalty functions (eq 4.20-4.21) - STRONG enforcement of T window
        penalty = 0.0
        rho = 5000.0
        if T12 < 70: penalty += rho*(70 - T12)**2
        if T12 > 76: penalty += rho*(T12 - 76)**2
        if mu < MU_MIN: penalty += rho*(MU_MIN - mu)**2
        if mu > MU_MAX: penalty += rho*(mu - MU_MAX)**2
        if Pp < PP_MIN: penalty += rho*((PP_MIN - Pp)/1e5)**2
        if Pp > PP_MAX: penalty += rho*((Pp - PP_MAX)/1e5)**2
        if Gv < GV_MIN: penalty += rho*(GV_MIN - Gv)**2
        if Gv > GV_MAX: penalty += rho*(Gv - GV_MAX)**2

        return J + penalty

    def _optimize_upper(self, Kc):
        """Upper level: Nelder-Mead optimization from current u_opt."""
        res = minimize(self._objective, self.u_opt, args=(Kc,),
                       method="Nelder-Mead",
                       options={"maxiter": 60, "xatol": 1e-4, "fatol": 1e-3})
        return res.x

    def act(self, obs, info):
        self.step_count += 1
        Kc_measured = info.get("Kc", 1.0)

        # Step 3-4: measure disturbance deviation
        delta_d = abs(Kc_measured - self.Kc_nominal)

        # Upper level: periodic recompute (low-frequency compensation)
        if self.step_count % self.recompute_every == 1:
            self.u_opt = self._optimize_upper(Kc_measured)
            self.Kc_nominal = Kc_measured

        # Lower level: if deviation exceeds threshold, apply correction
        u_applied = self.u_opt.copy()
        if delta_d > self.threshold:
            # Simplified lower-level correction: adjust mu proportionally
            # to compensate the disturbance effect on temperature
            correction = -0.5 * (Kc_measured - self.Kc_nominal)
            u_applied[0] = np.clip(u_applied[0] + correction, MU_MIN, MU_MAX)

        mu, Pp, Gv = u_applied
        mu = np.clip(mu, MU_MIN, MU_MAX)
        Pp = np.clip(Pp, PP_MIN, PP_MAX)
        Gv = np.clip(Gv, GV_MIN, GV_MAX)

        return np.array([norm_mu(mu), norm_pp(Pp), norm_gv(Gv)],
                        dtype=np.float32)


# ============================================================
# MORL-SAC wrapper (loads trained preference-conditioned actor)
# ============================================================
class MORLAgent:
    """Wrapper to use a trained preference-conditioned MORL-SAC actor as a
    controller. The desired preference w is supplied at construction and is
    appended to the observation, so the same trained model can realize any
    operating point on the Pareto front."""

    def __init__(self, actor_path, obs_dim=43, act_dim=3, hidden=256,
                 preference=(0.5, 0.25, 0.25)):
        import torch
        from train_morl_sac import Actor
        self.torch = torch
        self.actor = Actor(obs_dim, act_dim, hidden)
        self.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        self.actor.eval()
        w = np.asarray(preference, dtype=np.float32)
        self.preference = w / w.sum()

    def reset(self):
        pass

    def act(self, obs, info):
        import torch
        # The environment already appends the active preference to obs, but we
        # overwrite the last 3 entries to guarantee the requested preference.
        obs = np.asarray(obs, dtype=np.float32).copy()
        obs[-3:] = self.preference
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mu, _ = self.actor(o)
            a = torch.tanh(mu)
        return a.numpy()[0]


if __name__ == "__main__":
    print("Baseline controllers module loaded successfully.")
    print("Available controllers:")
    print("  1. PIDController")
    print("  2. MPCController")
    print("  3. FuzzyController")
    print("  4. TwoLevelHierarchical (previous PhD method)")
    print("  5. MORLAgent (trained MORL-SAC)")
