# Drone-Vortex-Ring-Simulation — modeling assumptions (AIAA honesty lock)

## Named physics

- **Kelvin-type scaling:** \(\Gamma \sim \sqrt{T \cdot 4\pi R / \rho}\) (actuator-disk inspired).
- **Helmholtz thin-ring self-induction:** \(U = \frac{\Gamma}{4\pi R}\big(\ln(8R/a) - 1/4\big)\).
- **Viscous decay:** \(\Gamma\) decays with effective viscosity and core size; core grows diffusively.

## Explicit non-claims

- Not a Navier–Stokes / LES / RANS CFD solver.
- Ring–ring interaction in the GUI is **heuristic damping**, not full Biot–Savart mutual induction.
- Regime labels (clean / interfering wake) are qualitative UI aids.
- IISc connection: 10-day May 2025 internship related to vortex tracking; **these CFD/reduced-order codes are self-driven personal work**.

## V&V checks for AIAA

- Closed-form Helmholtz identity vs `VortexRingSimple.self_induced_speed`.
- Exact Kelvin \(\Gamma \leftrightarrow T\) invertibility.
- Frozen-core exponential decay vs analytical \(\Gamma(t)\).
- Headless step-thrust wake export for reproducibility.
