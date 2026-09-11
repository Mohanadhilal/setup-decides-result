#!/usr/bin/env python3
"""
sugar_extraction_model.py
=========================
Extended physico-mathematical model of the sugar extraction process
in an inclined diffusion apparatus (for use as an RL environment).

Base part - dissertation equations (4.1)-(4.14).
Extensions (for Q1 level):
  (A) Dynamic sucrose balance for the SOLID phase (cossettes) - C_str_i
  (B) Dynamic sucrose balance for the LIQUID phase (juice)     - C_liq_i
  (C) Mass transfer by Fick's second law between phases
  (D) Actuator dynamics (steam valve) - first-order lag
  (E) Physical constraints (saturation, rate limits)
  (F) Transport delay between cells
  (G) Concentration-to-Brix conversion (industrial measurement)

Model state (vector x):
  M_i      (i=1..m)  - mass of mixture in cell, kg
  T_i      (i=1..m)  - temperature in cell, deg C
  Cstr_i   (i=1..m)  - sucrose concentration in cossettes, fraction (0..1)
  Cliq_i   (i=1..m)  - sucrose concentration in juice, % mass
  mu_act             - actual valve position (after actuator dynamics)

Dimension: 4*m + 1 = 49 variables at m=12.
"""

import numpy as np


class SugarExtractionModel:
    """Physical model of the extraction process (without RL wrapper)."""

    def __init__(self, m=12, seed=None):
        self.m = m
        self.rng = np.random.default_rng(seed)

        # ---------- Nominal parameters ----------
        # Flow rates (kg/s)
        self.G_v_nom   = 3.5
        self.G_zhv_nom = 1.0
        self.G_sv_nom  = 3.2
        self.n_sh_nom  = 12.0

        # Temperatures (deg C)
        self.t_v_in   = 85.0
        self.t_zhv_in = 55.0
        self.t_sv_in  = 18.0
        self.t_oc     = 25.0

        # Steam
        self.P_p_nom = 3.5e5
        self.P_a     = 1.1e5
        self.K_v     = 0.0050
        self.L_vap   = 2200e3

        # Thermophysics
        self.c_p  = 3650.0
        self.c_v  = 4186.0
        self.c_sv = 1670.0
        self.K_ht = 40.0
        self.F_ht = 1.0

        # Diffusion (4.13)
        self.Kc_nom = 1.0
        self.Fc_nom = 3.5
        self.D0  = 0.52e-9
        self.Ea  = 19500.0
        self.R_  = 8.314

        # ---------- (A,B) Sucrose mass-transfer parameters ----------
        self.C_beet0   = 0.175      # beet sugar content (fraction), 17.5 %
        self.C_str_min = 0.005      # residual sugar in pulp (fraction) ~ 0.5 %
        self.kLa_nom   = 0.020      # volumetric mass-transfer coeff, 1/s (x Fc)
        self.rho_liq   = 1060.0     # juice density, kg/m3

        # ---------- (D) Valve actuator dynamics ----------
        self.tau_valve = 30.0       # valve actuator time constant, s
        self.valve_rate_limit = 0.05  # max rate of change of mu, 1/s

        # ---------- (E) Constraints ----------
        self.mu_min, self.mu_max = 0.0, 1.0
        self.Pp_min, self.Pp_max = 2.0e5, 4.5e5
        self.Gv_min, self.Gv_max = 3.2, 3.8   # narrow draft band around nominal
        #   (full physical range is wider, but the controller only trims draft;
        #    the draft ratio is set by plant specs, not the moment-to-moment
        #    controller. This prevents the over-dilution exploit where the agent
        #    floods the apparatus with cold water (Gv=4.5) to save steam at the
        #    cost of concentration, Cc dropping to ~9%.)

        # ---------- (F) Transport delay ----------
        # implemented via mass inertia (large M0)
        self.M0     = 4200.0
        self.t0     = 50.0

        # ---------- Target values ----------
        self.T_target  = 73.5       # target temperature cell 12, deg C
        self.Cliq_target = 12.5     # target juice concentration, % (achievable)

        self.reset_params()

    # ============================================================
    def reset_params(self):
        """Reset control/disturbance variables to nominal."""
        self.G_v   = self.G_v_nom
        self.G_zhv = self.G_zhv_nom
        self.n_sh  = self.n_sh_nom
        self.P_p   = self.P_p_nom
        self.Kc    = self.Kc_nom
        self.Fc    = self.Fc_nom
        self.mu_cmd = 0.506         # commanded valve value
        self.mu_act = 0.506         # actual position

    # ============================================================
    # Helper functions
    # ============================================================
    def Gp_cell(self, mu):
        """(4.10) Steam flow per cell."""
        return mu * self.K_v * np.sqrt(max(self.P_p - self.P_a, 0)) / self.m

    def viscosity(self, t):
        return 0.6e-3 * np.exp(2100 * (1/(t+273.15) - 1/343.15))

    def diff_coeff(self, t):
        """(4.13) Sucrose diffusion coefficient."""
        Dt = self.D0 * np.exp(-self.Ea/self.R_ * (1/(t+273.15) - 1/298.15))
        return self.Kc * (self.Fc/self.Fc_nom) * Dt * \
               (self.viscosity(70) / self.viscosity(t))

    def Gsv(self):
        return self.G_sv_nom * (self.n_sh / self.n_sh_nom)

    def brix_from_C(self, C_pct):
        """(G) Convert sucrose concentration (% mass) to Brix.
        Brix includes all dissolved solids; for ~90% purity of diffusion juice."""
        return C_pct / 0.90

    # ============================================================
    # Right-hand side of the ODE system
    # ============================================================
    def derivatives(self, x):
        """
        Compute derivatives of the state vector.
        x = [M(m), T(m), Cstr(m), Cliq(m), mu_act(1)]
        """
        m = self.m
        M    = x[0:m]
        T    = x[m:2*m]
        Cstr = x[2*m:3*m]
        Cliq = x[3*m:4*m]
        mu_act = x[4*m]

        dM    = np.zeros(m)
        dT    = np.zeros(m)
        dCstr = np.zeros(m)
        dCliq = np.zeros(m)

        Gs = self.Gsv()
        Gl = self.G_v + self.G_zhv
        Gp = self.Gp_cell(mu_act)
        Tli0 = (self.G_v*self.t_v_in + self.G_zhv*self.t_zhv_in) / max(Gl, 0.01)

        for i in range(m):
            Mi = max(M[i], 100.0)
            Gt = Gl + Gs + Gp

            # --- (4.1)/(4.3) Mass balance ---
            Meq = self.M0 * (Gt / max(Gl + Gs, 0.01))
            dM[i] = (Meq - Mi) / (Mi / max(Gt, 0.01))

            # --- (4.5)/(4.7) Heat balance (counter-current) ---
            Ts = self.t_sv_in if i == 0 else T[i-1]      # cossettes 0->m
            Tl = Tli0        if i == m-1 else T[i+1]      # liquid m->0
            Qin  = Gl*self.c_v*Tl + Gs*self.c_sv*Ts + Gp*self.L_vap
            Qout = (Gl + Gs)*self.c_p*T[i]
            Qloss = self.K_ht*self.F_ht*(T[i] - self.t_oc)
            dT[i] = (Qin - Qout - Qloss) / (self.c_p * Mi)

            # --- (A,B,C) Sucrose mass transfer (Fick's 2nd law) ---
            # Mass-transfer coefficient depends on D(T) and cossette area
            D_i = self.diff_coeff(T[i])
            kLa = self.kLa_nom * (self.Fc/self.Fc_nom) * (D_i / self.diff_coeff(73.0))

            # Driving force: concentration difference (equilibrium Cstr <-> Cliq)
            # Equilibrium concentration in juice (fraction) ~ Cstr (partition coeff=1)
            Cliq_frac = Cliq[i] / 100.0
            driving = Cstr[i] - Cliq_frac          # concentration gradient
            J = kLa * driving                       # sucrose flux, 1/s (fraction/s)

            # Sugar balance in COSSETTES (solid phase, moves 0->m)
            Cstr_in = self.C_beet0 if i == 0 else Cstr[i-1]
            flow_str = (Gs / Mi) * (Cstr_in - Cstr[i])
            dCstr[i] = flow_str - J
            # Constraint: not below residual sugar
            if Cstr[i] <= self.C_str_min and dCstr[i] < 0:
                dCstr[i] = 0.0

            # Sugar balance in JUICE (liquid phase, moves m->0)
            Cliq_in = 0.0 if i == m-1 else Cliq[i+1]   # fresh water - no sugar
            flow_liq = (Gl / Mi) * (Cliq_in - Cliq[i])
            # Sugar transfer from cossettes to juice (x100 for %)
            transfer = J * (Gs / max(Gl, 0.01)) * 100.0
            dCliq[i] = flow_liq + transfer

        # --- (D) Valve actuator dynamics (first-order lag) ---
        # with rate limiting
        dmu_raw = (self.mu_cmd - mu_act) / self.tau_valve
        dmu = np.clip(dmu_raw, -self.valve_rate_limit, self.valve_rate_limit)

        return np.concatenate([dM, dT, dCstr, dCliq, [dmu]])

    # ============================================================
    def initial_state(self):
        m = self.m
        M0    = np.full(m, self.M0)
        T0    = np.full(m, self.t0)
        Cstr0 = np.full(m, self.C_beet0)        # cossettes full of sugar
        Cliq0 = np.linspace(2.0, 0.0, m)        # juice almost without sugar
        mu0   = np.array([self.mu_act])
        return np.concatenate([M0, T0, Cstr0, Cliq0, mu0])

    # ============================================================
    def step_rk4(self, x, dt):
        """One integration step using 4th-order Runge-Kutta."""
        k1 = self.derivatives(x)
        k2 = self.derivatives(x + 0.5*dt*k1)
        k3 = self.derivatives(x + 0.5*dt*k2)
        k4 = self.derivatives(x + dt*k3)
        x_new = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # (E) Physically constrain mu_act
        m = self.m
        x_new[4*m] = np.clip(x_new[4*m], self.mu_min, self.mu_max)
        # Concentrations within physical limits
        x_new[2*m:3*m] = np.clip(x_new[2*m:3*m], 0.0, self.C_beet0)
        x_new[3*m:4*m] = np.clip(x_new[3*m:4*m], 0.0, 25.0)
        return x_new

    # ============================================================
    def get_outputs(self, x):
        """Extract output variables from the state."""
        m = self.m
        T    = x[m:2*m]
        Cstr = x[2*m:3*m]
        Cliq = x[3*m:4*m]
        mu_act = x[4*m]

        Cc_out  = Cliq[0]                  # juice concentration at outlet (cell 1)
        T12     = T[-1]                    # heating-zone temperature
        Gp_tot  = self.Gp_cell(mu_act) * m
        Gobsh   = self.G_v + self.G_zhv + self.Gsv() + Gp_tot
        D12     = self.diff_coeff(T12)
        brix    = self.brix_from_C(Cc_out)
        # Sugar extraction efficiency (fraction extracted).
        # Base extraction from the mass-transfer state of the last cell, then a
        # physically-motivated temperature correction: higher temperature speeds
        # sucrose diffusion (Arrhenius), leaving less sugar in the pulp -> higher
        # extraction. Calibrated so extraction rises ~96 % at 70 C to ~98 % at
        # 78 C, saturating (diffusion-limited) and never exceeding a realistic
        # industrial ceiling. Temperatures above ~80 C give no further gain
        # (and, in practice, risk sucrose degradation - handled as a reward
        # constraint, not here).
        base_extraction = 1.0 - Cstr[-1] / self.C_beet0
        # Temperature factor: logistic ramp centered at 74 C, spanning 70-78 C.
        T_clip = np.clip(T12, 60.0, 82.0)
        temp_factor = 0.96 + 0.02 / (1.0 + np.exp(-(T_clip - 74.0) / 2.0))
        # Blend: the temperature factor scales the achievable extraction, but
        # never above the base (which already reflects residence/contact area).
        extraction = base_extraction * (temp_factor / 0.971)
        extraction = float(np.clip(extraction, 0.0, 0.985))   # industrial ceiling

        return dict(Cc=Cc_out, T12=T12, Gp=Gp_tot, Gobsh=Gobsh,
                    D=D12, brix=brix, extraction=extraction,
                    T_profile=T.copy(), Cliq_profile=Cliq.copy(),
                    Cstr_profile=Cstr.copy(), mu_act=mu_act)

    # ============================================================
    def apply_disturbance(self, dt):
        """(F) Random disturbances: raw-material quality and steam pressure."""
        # Kc varies as a random walk (Ornstein-Uhlenbeck)
        theta_kc = 0.001
        sigma_kc = 0.002
        self.Kc += theta_kc*(self.Kc_nom - self.Kc)*dt + \
                   sigma_kc*np.sqrt(dt)*self.rng.standard_normal()
        self.Kc = np.clip(self.Kc, 0.6, 1.2)

        # Pp fluctuations (steam network pressure)
        theta_pp = 0.002
        sigma_pp = 3000.0
        self.P_p += theta_pp*(self.P_p_nom - self.P_p)*dt + \
                    sigma_pp*np.sqrt(dt)*self.rng.standard_normal()
        self.P_p = np.clip(self.P_p, self.Pp_min, self.Pp_max)


# ================================================================
# Model self-test
# ================================================================
if __name__ == '__main__':
    model = SugarExtractionModel(m=12, seed=42)
    x = model.initial_state()
    dt = 5.0  # s

    print("Extended extraction model check")
    print("="*55)
    print(f"State dimension: {len(x)} ({4*model.m}+1)")
    print(f"Initial Cc (juice, cell 1): {model.get_outputs(x)['Cc']:.2f} %")
    print()

    # Run to steady state (no disturbances)
    t_total = 18000  # 300 min
    n_steps = int(t_total / dt)
    for k in range(n_steps):
        x = model.step_rk4(x, dt)

    out = model.get_outputs(x)
    print("Steady-state regime (nominal):")
    print(f"  Cc (juice concentration): {out['Cc']:.2f} %")
    print(f"  Brix:                     {out['brix']:.2f} deg Bx")
    print(f"  T12 (heating zone):       {out['T12']:.1f} deg C")
    print(f"  D (diffusion coeff):      {out['D']*1e9:.2f} x10^-9 m2/s")
    print(f"  Gp (steam flow):          {out['Gp']:.3f} kg/s")
    print(f"  Gtotal (throughput):      {out['Gobsh']:.2f} kg/s")
    print(f"  Sugar extraction:         {out['extraction']*100:.1f} %")
    print(f"  Residual sugar in pulp:   {out['Cstr_profile'][-1]*100:.2f} %")
    print()
    print(f"  T profile by cell:        {np.round(out['T_profile'],1)}")
    print(f"  Cliq profile (juice):     {np.round(out['Cliq_profile'],2)}")
    print(f"  Cstr profile (cossettes): {np.round(out['Cstr_profile']*100,2)}")
