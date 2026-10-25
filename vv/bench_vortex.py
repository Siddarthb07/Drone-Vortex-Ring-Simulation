"""
V&V harness for reduced-order vortex-ring model.

Checks:
1) Helmholtz U formula identity (code vs closed form)
2) Kelvin Γ ↔ T invertibility RMSE
3) Frozen-core exponential decay vs analytical Γ(t)
4) Headless wake run under step thrust (artifact only)

Usage:
  python -m vv.bench_vortex
  python vv/bench_vortex.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vortex_physics import (  # noqa: E402
    NU_AIR,
    RHO_AIR,
    VortexRingSimple,
    circulation_to_thrust,
    helmholtz_self_induced_speed,
    thrust_profile,
    thrust_to_circulation,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run():
    # --- Helmholtz identity ---
    Gamma_grid = np.linspace(0.5, 8.0, 16)
    R = 0.15
    a0 = 0.01
    U_code = []
    U_closed = []
    for G in Gamma_grid:
        ring = VortexRingSimple(R, z0=0.0, Gamma=G, nu_eff=NU_AIR)
        ring.a = a0
        U_code.append(ring.self_induced_speed())
        U_closed.append(helmholtz_self_induced_speed(G, R, a0))
    u_rmse = rmse(U_code, U_closed)

    # --- Kelvin invertibility ---
    T_grid = np.linspace(1.0, 30.0, 30)
    T_back = [circulation_to_thrust(thrust_to_circulation(T, R), R) for T in T_grid]
    kelvin_rmse = rmse(T_grid, T_back)
    kelvin_max_abs_pct = float(
        np.max(100.0 * np.abs(np.asarray(T_back) - T_grid) / T_grid)
    )

    # --- Frozen-core decay (hold a fixed; compare to Γ0 exp(-ν t / a^2)) ---
    Gamma0 = 5.0
    nu = NU_AIR
    a_fixed = 0.01
    dt = 0.01
    t_end = 2.0
    times = np.arange(0.0, t_end + dt, dt)
    ring = VortexRingSimple(R, z0=0.0, Gamma=Gamma0, nu_eff=nu)
    ring.a = a_fixed
    gamma_num = []
    gamma_ana = []
    for t in times:
        # analytical assumes frozen a
        gamma_ana.append(Gamma0 * np.exp(-nu * t / (a_fixed**2)))
        gamma_num.append(ring.Gamma)
        # custom step with frozen core (do not grow a)
        ring.Gamma *= np.exp(-nu * dt / (ring.a**2 + 1e-9))
        U = ring.self_induced_speed()
        ring.z -= U * dt
    decay_rmse = rmse(gamma_num, gamma_ana)

    # --- Headless wake under step thrust ---
    T_max = 15.0
    dt_w = 0.03
    t = 0.0
    last_emit = -999.0
    rings = []
    hist_t, hist_g = [], []
    while t < 3.0:
        t += dt_w
        if t - last_emit >= 0.12:
            T_now = thrust_profile(t, "step", T_max)
            if T_now > 0.1:
                G = thrust_to_circulation(T_now, R, RHO_AIR)
                rings.append(VortexRingSimple(R, z0=-0.02, Gamma=G, nu_eff=NU_AIR))
                last_emit = t
                if len(rings) > 25:
                    rings.pop(0)
        for r in rings:
            r.update(dt_w)
        hist_t.append(t)
        hist_g.append(sum(max(r.Gamma, 0.0) for r in rings))

    wake_csv = OUT / "wake_step_gamma.csv"
    with wake_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "total_circulation_m2_per_s"])
        for ti, gi in zip(hist_t, hist_g):
            w.writerow([ti, gi])

    summary = {
        "helmholtz_U_rmse_m_s": u_rmse,
        "kelvin_T_invert_rmse_N": kelvin_rmse,
        "kelvin_T_invert_max_abs_pct": kelvin_max_abs_pct,
        "frozen_core_decay_rmse_m2_s": decay_rmse,
        "wake_step_final_total_gamma": float(hist_g[-1]) if hist_g else None,
        "n_wake_samples": len(hist_t),
        "assumptions": [
            "Thin-ring Helmholtz self-induction; thick-core corrections omitted",
            "Kelvin Γ–T scaling is actuator-disk inspired, not Navier–Stokes",
            "Ring–ring interaction in the GUI is heuristic damping, not Biot–Savart",
            "Not a CFD solver; educational / reduced-order research software",
        ],
    }
    (OUT / "vortex_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Error table CSV
    err_csv = OUT / "vortex_error.csv"
    with err_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "metric", "value"])
        w.writerow(["helmholtz_identity", "rmse_m_s", u_rmse])
        w.writerow(["kelvin_invertibility", "rmse_N", kelvin_rmse])
        w.writerow(["kelvin_invertibility", "max_abs_pct", kelvin_max_abs_pct])
        w.writerow(["frozen_core_decay", "rmse_m2_s", decay_rmse])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(Gamma_grid, U_closed, "k-", label="closed-form Helmholtz U")
    ax[0].plot(Gamma_grid, U_code, "o", label="VortexRingSimple")
    ax[0].set_xlabel("Γ [m²/s]")
    ax[0].set_ylabel("U [m/s]")
    ax[0].set_title("Self-induced speed identity")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(times, gamma_ana, "k-", label="analytical frozen-a")
    ax[1].plot(times, gamma_num, "--", label="numerical frozen-a")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("Γ [m²/s]")
    ax[1].set_title("Viscous decay check")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_vortex_vv.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist_t, hist_g)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("ΣΓ [m²/s]")
    ax.set_title("Step-thrust wake: total circulation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_wake_step.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {err_csv}")
    return summary


if __name__ == "__main__":
    run()
