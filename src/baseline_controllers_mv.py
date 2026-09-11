#!/usr/bin/env python3
"""
baseline_controllers_mv.py
==========================
Multivariable versions of the classical baselines, so that the comparison
with the learned controller is between control strategies rather than
between different numbers of manipulated variables.

Why these exist
---------------
In baseline_controllers.py every classical controller regulates the
heating-zone temperature by moving the steam valve alone, and holds the
supply pressure and the draft at nominal values; the docstrings say so
explicitly. The learned controller moves all three. That is one degree of
freedom against three, and it invalidates the comparisons that matter most:

  Juice concentration is set mainly by the draft. With the draft frozen the
  classical controllers cannot move it at all, which is why all three report
  a concentration of 12.31 percent, the value implied by Gv = 3.5. That
  number is not a control achievement, it is an arithmetic consequence of
  the constraint they were given.

  Plant-level energy depends on the concentration through the evaporation
  duty, so a comparison at equal plant energy was comparing a controller
  that can trade against one that cannot.

  The Pareto front itself is traced largely by the draft.

Any advantage measured that way is an advantage in degrees of freedom, not
in control strategy, and a reviewer will say so.

What is added
-------------
Each controller gains a second loop that regulates juice concentration by
moving the draft, which is standard industrial practice: draft control is a
normal part of diffuser operation, not an exotic extra. The loops are
deliberately conventional and detuned relative to the temperature loop, the
usual arrangement when an inner thermal loop and an outer quality loop share
a plant, so that the baselines remain realistic rather than strawmen tuned
to fail.

The supply pressure remains at nominal for the classical controllers. It is
a utility variable that plants do not normally manipulate for control, so
freezing it is defensible; state that choice in the paper rather than
leaving it implicit. If the learned controller is found to gain mainly by
moving the supply pressure, that should be reported, and the honest response
would be to freeze it for the learned controller too.

Tuning
------
All gains are listed as constructor defaults so they can be reported in a
parameter table. They were set by the standard rule of making the outer
concentration loop roughly five times slower than the inner temperature
loop, then checked for absence of oscillation; they were not searched over
to obtain any particular comparison outcome.

Usage
-----
    from baseline_controllers_mv import (PIDControllerMV, MPCControllerMV,
                                         FuzzyControllerMV)
    ctrl = PIDControllerMV(dt=env.dt)

The interface matches the original module exactly.
"""

import numpy as np
from scipy.optimize import minimize

from baseline_controllers import (MU_MIN, MU_MAX, PP_MIN, PP_MAX,
                                  GV_MIN, GV_MAX, norm_mu, norm_pp, norm_gv,
                                  MPCController)


# ============================================================
# Shared outer loop: juice concentration -> draft
# ============================================================
class DraftLoop:
    """PI loop regulating juice concentration Cc by moving the draft Gv.

    Raising the draft adds extraction water, which dilutes the juice, so the
    process gain from Gv to Cc is negative and the loop subtracts its output.
    The integral term is clamped and the output saturated at the physical
    draft limits, so the loop cannot wind up when the concentration target is
    unreachable.

    The default gains make this loop about five times slower than the
    temperature loop, which is the conventional arrangement when a quality
    loop sits outside a thermal one; a comparably fast outer loop would
    fight the inner one.
    """

    def __init__(self, Cc_target=12.30, Kp=0.05, Ki=0.0025,
                 Gv_bias=3.5, dt=10.0, i_clip=200.0):
        self.Cc_target = Cc_target
        self.Kp, self.Ki = Kp, Ki
        self.Gv_bias = Gv_bias
        self.dt = dt
        self.i_clip = i_clip
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def __call__(self, Cc):
        err = self.Cc_target - Cc          # positive when juice is too weak
        self.integral = float(np.clip(self.integral + err * self.dt,
                                      -self.i_clip, self.i_clip))
        # negative sign: more draft dilutes, so a positive error calls for
        # less water, not more
        gv = self.Gv_bias - (self.Kp * err + self.Ki * self.integral)
        return float(np.clip(gv, GV_MIN, GV_MAX))


# ============================================================
# 1. PID with a draft loop
# ============================================================
class PIDControllerMV:
    """PID on temperature through the steam valve, plus a PI draft loop on
    juice concentration. Supply pressure held at nominal."""

    def __init__(self, T_target=73.5, Kp=0.04, Ki=0.002, Kd=0.01,
                 Cc_target=12.30, Kp_gv=0.05, Ki_gv=0.0025,
                 Pp_fixed=3.5e5, Gv_bias=3.5, dt=10.0):
        self.T_target = T_target
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Pp_fixed = Pp_fixed
        self.dt = dt
        self.integral = 0.0
        self.prev_err = 0.0
        self.draft = DraftLoop(Cc_target, Kp_gv, Ki_gv, Gv_bias, dt)

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.draft.reset()

    def act(self, obs, info):
        T12 = info["T12"]
        err = self.T_target - T12
        self.integral = float(np.clip(self.integral + err * self.dt, -500, 500))
        deriv = (err - self.prev_err) / self.dt
        self.prev_err = err

        mu = 0.5 + self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        mu = float(np.clip(mu, MU_MIN, MU_MAX))
        gv = self.draft(info["Cc"])

        return np.array([norm_mu(mu), norm_pp(self.Pp_fixed), norm_gv(gv)],
                        dtype=np.float32)


# ============================================================
# 2. MPC with a draft loop
# ============================================================
class MPCControllerMV:
    """Receding-horizon MPC on temperature through the steam valve, with the
    same PI draft loop layered outside it.

    The draft is not placed inside the horizon on purpose. Doing so would
    turn the baseline into a small multivariable economic optimizer, which is
    a different and stronger method than the single-loop MPC the paper is
    comparing against, and it would blur the distinction the comparison is
    meant to draw. An economic multivariable NMPC remains the right further
    baseline, and its absence is stated as a limitation.
    """

    def __init__(self, model, T_target=73.5, Cc_target=12.30,
                 Kp_gv=0.05, Ki_gv=0.0025, Pp_fixed=3.5e5, Gv_bias=3.5,
                 dt=10.0, **kwargs):
        self._inner = MPCController(model, T_target=T_target,
                                    Pp_fixed=Pp_fixed, Gv_fixed=Gv_bias,
                                    dt=dt, **kwargs)
        self.draft = DraftLoop(Cc_target, Kp_gv, Ki_gv, Gv_bias, dt)
        self.Pp_fixed = Pp_fixed

    def reset(self):
        if hasattr(self._inner, "reset"):
            self._inner.reset()
        self.draft.reset()

    def act(self, obs, info):
        a = self._inner.act(obs, info)          # already normalized
        gv = self.draft(info["Cc"])
        a = np.asarray(a, dtype=np.float32).copy()
        a[2] = norm_gv(gv)                      # replace the frozen draft
        return a


# ============================================================
# 3. Fuzzy with a draft loop
# ============================================================
class FuzzyControllerMV:
    """Mamdani fuzzy controller on temperature through the steam valve, with
    the same PI draft loop on juice concentration.

    The fuzzy rule base is left untouched and only the frozen draft is
    replaced, so that any difference against the original baseline is
    attributable to the added degree of freedom and not to retuning.
    """

    def __init__(self, T_target=73.5, Cc_target=12.30,
                 Kp_gv=0.05, Ki_gv=0.0025, Pp_fixed=3.5e5, Gv_bias=3.5,
                 dt=10.0):
        from baseline_controllers import FuzzyController
        self._inner = FuzzyController(T_target=T_target, Pp_fixed=Pp_fixed,
                                      Gv_fixed=Gv_bias, dt=dt)
        self.draft = DraftLoop(Cc_target, Kp_gv, Ki_gv, Gv_bias, dt)

    def reset(self):
        if hasattr(self._inner, "reset"):
            self._inner.reset()
        self.draft.reset()

    def act(self, obs, info):
        a = np.asarray(self._inner.act(obs, info), dtype=np.float32).copy()
        a[2] = norm_gv(self.draft(info["Cc"]))
        return a


# ============================================================
DOF_TABLE = """
Manipulated variables, for the paper's methods section
------------------------------------------------------
  controller            steam valve   supply pressure   draft
  PID (original)            yes           nominal       nominal
  MPC (original)            yes           nominal       nominal
  Fuzzy (original)          yes           nominal       nominal
  PID / MPC / Fuzzy MV      yes           nominal        yes
  MORL-SAC                  yes             yes          yes

The supply pressure remains at nominal for every classical controller. It is
a utility variable rather than a normal handle for diffuser control, so this
is defensible, but it does leave the learned controller with one extra
degree of freedom. If the learned advantage is found to depend on moving the
supply pressure, freeze it for the learned controller too and report both.
"""

if __name__ == "__main__":
    print(__doc__.split("Usage")[0])
    print(DOF_TABLE)
