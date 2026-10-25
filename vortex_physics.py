"""
Headless vortex-ring physics (no Matplotlib UI).

Supports arbitrary thrust-axis unit vectors for multi-rotor cinema.
Legacy GUI path (no n_hat) keeps z decreasing with self-induction.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

RHO_AIR = 1.225
NU_AIR = 1.5e-5


def thrust_profile(t: float, mode: str, T_max: float) -> float:
    if mode == "step":
        return T_max if t > 0.2 else 0.0
    if mode == "slow":
        return float(np.clip(T_max * t / 1.0, 0.0, T_max))
    if mode == "fast":
        return float(np.clip(T_max * t / 0.25, 0.0, T_max))
    if mode == "hoverd":
        if t < 1.0:
            return T_max
        return T_max * (0.6 + 0.2 * np.sin(2 * np.pi * (t - 1.0)))
    if mode == "pulses":
        period = 0.8
        phase = (t % period) / period
        return T_max if phase < 0.4 else 0.2 * T_max
    return T_max


def thrust_to_circulation(T: float, R: float, rho: float = RHO_AIR) -> float:
    T = max(T, 0.0)
    return float(np.sqrt(T * 4.0 * np.pi * R / max(rho, 1e-9)))


def circulation_to_thrust(Gamma: float, R: float, rho: float = RHO_AIR) -> float:
    return float((Gamma**2) * rho / (4.0 * np.pi * R))


def helmholtz_self_induced_speed(Gamma: float, R: float, a: float) -> float:
    R = max(R, 1e-4)
    a = max(a, 1e-4)
    return float(Gamma / (4.0 * np.pi * R) * (np.log(8.0 * R / a) - 0.25))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, -1.0])
    return v / n


class VortexRingSimple:
    def __init__(
        self,
        R: float,
        z0: float,
        Gamma: float,
        nu_eff: float,
        origin: Optional[Sequence[float]] = None,
        n_hat: Optional[Sequence[float]] = None,
        rotor_id: int = 0,
    ):
        self.R = R
        self.Gamma = Gamma
        self.a = 0.01
        self.nu_eff = nu_eff
        self.rotor_id = rotor_id
        self._legacy = origin is None and n_hat is None
        if self._legacy:
            self.z = float(z0)
            self.origin = np.zeros(3)
            self.n_hat = np.array([0.0, 0.0, -1.0])
            self.pos = np.array([0.0, 0.0, self.z])
        else:
            self.origin = np.asarray(origin if origin is not None else [0, 0, 0], dtype=float)
            self.n_hat = _unit(n_hat if n_hat is not None else [0.0, 0.0, -1.0])
            # z0 interpreted as initial distance along wake axis
            self.s = abs(float(z0)) if float(z0) != 0 else 0.02
            self.pos = self.origin + self.s * self.n_hat
            self.z = float(self.pos[2])

    def self_induced_speed(self) -> float:
        return helmholtz_self_induced_speed(self.Gamma, self.R, self.a)

    def update(self, dt: float):
        self.a = max(self.a, 1e-4)
        self.a += np.sqrt(4 * self.nu_eff * dt) * 0.2
        self.Gamma *= np.exp(-self.nu_eff * dt / (self.a**2 + 1e-9))
        U = self.self_induced_speed()
        if self._legacy:
            self.z -= U * dt
            self.pos = np.array([0.0, 0.0, self.z])
        else:
            self.s += U * dt
            self.pos = self.origin + self.s * self.n_hat
            self.z = float(self.pos[2])

    def weaken_due_to_interaction(self, factor: float):
        self.Gamma *= factor

    def state_tuple(self) -> Tuple:
        return (
            self.rotor_id,
            float(self.Gamma),
            float(self.R),
            float(self.a),
            float(self.pos[0]),
            float(self.pos[1]),
            float(self.pos[2]),
            float(self.self_induced_speed()),
            float(self.n_hat[0]),
            float(self.n_hat[1]),
            float(self.n_hat[2]),
        )


class MultiRotorWake:
    def __init__(self, nu_scale: float = 1.0, emit_dt: float = 0.12, max_rings: int = 80):
        self.nu_eff = NU_AIR * nu_scale
        self.emit_dt = emit_dt
        self.max_rings = max_rings
        self.rings: List[VortexRingSimple] = []
        self.last_emit: dict = {}

    def maybe_emit(
        self,
        t: float,
        rotor_id: int,
        thrust: float,
        radius: float,
        origin: Sequence[float],
        n_hat: Sequence[float],
    ):
        if t - self.last_emit.get(rotor_id, -999.0) < self.emit_dt:
            return
        if thrust <= 0.1:
            return
        Gamma = thrust_to_circulation(thrust, radius)
        ring = VortexRingSimple(
            R=radius,
            z0=0.02,
            Gamma=Gamma,
            nu_eff=self.nu_eff,
            origin=origin,
            n_hat=n_hat,
            rotor_id=rotor_id,
        )
        self.rings.append(ring)
        self.last_emit[rotor_id] = t
        if len(self.rings) > self.max_rings:
            self.rings.pop(0)

    def step(self, dt: float):
        for r in self.rings:
            r.update(dt)
        for i in range(1, len(self.rings)):
            if np.linalg.norm(self.rings[i].pos - self.rings[i - 1].pos) < 0.05:
                self.rings[i].weaken_due_to_interaction(0.85)

    def snapshot(self):
        return [r.state_tuple() for r in self.rings]

    def total_gamma(self) -> float:
        return float(sum(max(r.Gamma, 0.0) for r in self.rings))
