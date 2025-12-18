import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import csv
import os

# Physical constants (SI)
RHO_AIR = 1.225       # kg/m^3
NU_AIR = 1.5e-5       # m^2/s  (base; slider scales this)
G = 9.81              # m/s^2 (for Reynolds scale)


# --------------------------------------------
# Thrust profiles
# --------------------------------------------

def thrust_profile(t, mode, T_max):
    """Thrust T(t) in Newtons."""
    if mode == "step":
        return T_max if t > 0.2 else 0.0
    elif mode == "slow":
        return np.clip(T_max * t / 1.0, 0.0, T_max)
    elif mode == "fast":
        return np.clip(T_max * t / 0.25, 0.0, T_max)
    elif mode == "hoverd":
        if t < 1.0:
            return T_max
        else:
            return T_max * (0.6 + 0.2 * np.sin(2 * np.pi * (t - 1.0)))
    elif mode == "pulses":
        period = 0.8
        phase = (t % period) / period
        return T_max if phase < 0.4 else 0.2 * T_max
    else:
        return T_max


# --------------------------------------------
# Thrust → circulation scaling
# --------------------------------------------

def thrust_to_circulation(T, R, rho):
    """
    Simple actuator-disc style scaling:
        T ~ ρ Γ^2 / (4πR)  =>  Γ ~ sqrt(T * 4πR / ρ)  (Kelvin-type scaling). [web:45]
    """
    T = max(T, 0.0)
    return np.sqrt(T * 4.0 * np.pi * R / max(rho, 1e-9))


# --------------------------------------------
# Ring object with Γ and viscous decay
# --------------------------------------------

class VortexRingSimple:
    """
    Each ring carries:
        Γ  : circulation [m^2/s]
        R  : ring radius [m]
        z  : vertical position [m]
        a  : core radius / thickness [m]

    Motion:
        U = Γ / (4πR) (ln(8R/a) - 1/4)  (Kelvin–Helmholtz thin-ring formula). [web:34][web:36][web:38]
    Decay:
        dΓ/dt = -ν_eff * Γ / a^2  (viscous diffusion scaling). [web:40][web:43][web:46]
    """

    def __init__(self, R, z0, Gamma, nu_eff):
        self.R = R
        self.z = z0
        self.Gamma = Gamma
        self.a = 0.01        # initial core size [m]
        self.nu_eff = nu_eff

    def update(self, dt):
        # grow core radius slowly (diffusive growth)
        self.a = max(self.a, 1e-4)
        self.a += np.sqrt(4 * self.nu_eff * dt) * 0.2

        # viscous circulation decay
        self.Gamma *= np.exp(-self.nu_eff * dt / (self.a**2 + 1e-9))

        # self-induced speed
        R = max(self.R, 1e-4)
        a = max(self.a, 1e-4)
        U = self.Gamma / (4.0 * np.pi * R) * (np.log(8.0 * R / a) - 0.25)  # m/s [web:34][web:36][web:38]

        # move downward with U (z decreases as ring travels away)
        self.z -= U * dt

    def weaken_due_to_interaction(self, factor):
        """Apply extra damping when rings get too close (simple interaction)."""
        self.Gamma *= factor


# --------------------------------------------
# Main simulation + UI
# --------------------------------------------

class DroneVortexApp:
    def __init__(self):
        # State
        self.mode = "step"
        self.dt = 0.03
        self.t = 0.0
        self.T_max = 15.0          # N
        self.rotor_radius = 0.15   # m
        self.visc_scale = 1.0      # multiplies NU_AIR
        self.rings = []
        self.last_emit_t = -999.0

        # Histories
        self.time_hist = []
        self.gamma_hist = []

        # Global comparison storage (for step/slow/fast overlay)
        self.compare_results = None

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 6))

        # Layout: left big 3D, right top info+dimensionless, right bottom graph
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_info = self.fig.add_subplot(222)
        self.ax_graph = self.fig.add_subplot(224)

        plt.subplots_adjust(left=0.06, right=0.80, bottom=0.20, top=0.92,
                            wspace=0.35, hspace=0.45)

        # Controls on right column
        axcolor = "0.1"
        self.ax_mode = plt.axes([0.82, 0.64, 0.17, 0.25], facecolor=axcolor)
        self.radio = RadioButtons(
            self.ax_mode,
            ("step", "slow", "fast", "hoverd", "pulses"),
            active=0
        )
        self.radio.on_clicked(self.change_mode)

        self.ax_T = plt.axes([0.82, 0.54, 0.16, 0.03], facecolor=axcolor)
        self.s_T = Slider(self.ax_T, "Max thrust [N]", 5, 30, valinit=self.T_max)

        self.ax_R = plt.axes([0.82, 0.49, 0.16, 0.03], facecolor=axcolor)
        self.s_R = Slider(self.ax_R, "Rotor radius [m]", 0.10, 0.30, valinit=self.rotor_radius)

        self.ax_visc = plt.axes([0.82, 0.44, 0.16, 0.03], facecolor=axcolor)
        self.s_visc = Slider(self.ax_visc, "ν scale [-]", 0.5, 3.0, valinit=self.visc_scale)

        self.ax_speed = plt.axes([0.82, 0.39, 0.16, 0.03], facecolor=axcolor)
        self.s_speed = Slider(self.ax_speed, "Time scale [-]", 0.5, 2.0, valinit=1.0)

        # Buttons: start/reset/save/compare
        self.ax_btn_start = plt.axes([0.82, 0.28, 0.07, 0.06])
        self.ax_btn_reset = plt.axes([0.91, 0.28, 0.07, 0.06])
        self.ax_btn_save = plt.axes([0.82, 0.18, 0.07, 0.06])
        self.ax_btn_comp = plt.axes([0.91, 0.18, 0.07, 0.06])

        self.btn_start = Button(self.ax_btn_start, "Pause")
        self.btn_reset = Button(self.ax_btn_reset, "Reset")
        self.btn_save = Button(self.ax_btn_save, "Save CSV")
        self.btn_comp = Button(self.ax_btn_comp, "Compare\nprofiles")

        self.btn_start.on_clicked(self.toggle_start)
        self.btn_reset.on_clicked(self.reset)
        self.btn_save.on_clicked(self.save_csv)
        self.btn_comp.on_clicked(self.run_comparison)

        self.running = True

        # Slider callbacks
        self.s_T.on_changed(self.update_params)
        self.s_R.on_changed(self.update_params)
        self.s_visc.on_changed(self.update_params)
        self.s_speed.on_changed(self.update_params)

        self.anim = FuncAnimation(self.fig, self.update, interval=30, blit=False)

    # ---------- UI callbacks ----------

    def change_mode(self, label):
        self.mode = label
        self.reset(None)

    def update_params(self, val):
        self.T_max = self.s_T.val
        self.rotor_radius = self.s_R.val
        self.visc_scale = self.s_visc.val
        self.dt = 0.03 * self.s_speed.val

    def toggle_start(self, event):
        self.running = not self.running
        self.btn_start.label.set_text("Start" if not self.running else "Pause")

    def reset(self, event):
        self.t = 0.0
        self.rings = []
        self.time_hist = []
        self.gamma_hist = []
        self.last_emit_t = -999.0

    def save_csv(self, event):
        if not self.time_hist:
            print("No data to save yet.")
            return

        filename = f"vortex_gamma_mode_{self.mode}.csv"
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filename):
            filename = f"{base}_{counter}{ext}"
            counter += 1

        with open(filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "total_circulation_m2_per_s"])
            for t, g in zip(self.time_hist, self.gamma_hist):
                w.writerow([t, g])
        print(f"Saved CSV: {filename}")

    # ---------- Simulation logic ----------

    def maybe_emit_ring(self):
        # emit new ring each ~0.12 s
        if self.t - self.last_emit_t < 0.12:
            return

        T_now = thrust_profile(self.t, self.mode, self.T_max)
        if T_now <= 0.1:
            return

        Gamma = thrust_to_circulation(T_now, self.rotor_radius, RHO_AIR)
        nu_eff = NU_AIR * self.visc_scale
        ring = VortexRingSimple(self.rotor_radius, z0=-0.02, Gamma=Gamma, nu_eff=nu_eff)
        self.rings.append(ring)
        self.last_emit_t = self.t

        if len(self.rings) > 25:
            self.rings.pop(0)

    def apply_ring_interactions(self):
        """
        Simple interaction: if two rings are too close vertically,
        reduce the circulation of the younger one to mimic interference / merging. [web:35][web:39][web:52]
        """
        if len(self.rings) < 2:
            return
        threshold = 0.05  # m
        for i in range(1, len(self.rings)):
            older = self.rings[i-1]
            newer = self.rings[i]
            if abs(newer.z - older.z) < threshold:
                newer.weaken_due_to_interaction(0.7)

    def step_sim(self):
        self.t += self.dt
        self.maybe_emit_ring()

        for r in self.rings:
            r.update(self.dt)

        self.apply_ring_interactions()

        total_gamma = sum(max(r.Gamma, 0.0) for r in self.rings)
        self.time_hist.append(self.t)
        self.gamma_hist.append(total_gamma)

    # ---------- Dimensionless numbers & regime ----------

    def compute_dimensionless(self):
        """
        Rough dimensionless quantities for display:
        - Re_Γ = Γ / ν  (circulation-based Reynolds). [web:35][web:43][web:52]
        - St ≈ f * R / U_ref: here approximate based on thrust pulse period. [web:39][web:42]
        """
        if not self.rings:
            return np.nan, np.nan, "No wake yet"

        nu_eff = NU_AIR * self.visc_scale
        # Take latest ring as representative
        r = self.rings[-1]
        Re_gamma = r.Gamma / max(nu_eff, 1e-9)

        # rough Strouhal-like parameter using pulse frequency
        if self.mode == "pulses":
            period = 0.8
            f = 1.0 / period
        else:
            f = 1.0  # arbitrary 1 Hz scale for others
        U_ref = max(abs(r.Gamma) / (2 * np.pi * r.R + 1e-9), 1e-3)
        St = f * r.R / U_ref

        # classify wake regime using variance of Γ and vertical spacing
        regime = "Clean wake"
        if len(self.gamma_hist) > 10:
            gamma_arr = np.array(self.gamma_hist[-30:])
            var_gamma = np.var(gamma_arr) / (np.mean(gamma_arr) + 1e-9)
        else:
            var_gamma = 0.0

        spacing = np.nan
        if len(self.rings) >= 2:
            spacing = abs(self.rings[-1].z - self.rings[-2].z)

        if np.isnan(spacing) or spacing > 0.08:
            if var_gamma < 0.2:
                regime = "Clean wake"
            else:
                regime = "Mildly unsteady wake"
        else:
            if var_gamma < 0.5:
                regime = "Interfering wake"
            else:
                regime = "Strongly unsteady wake"

        return Re_gamma, St, regime

    # ---------- Drawing ----------

    def draw_drone(self):
        body_size = 0.05
        xs = [-body_size, body_size, body_size, -body_size, -body_size]
        ys = [-body_size, -body_size, body_size, body_size, -body_size]
        zs = [0, 0, 0, 0, 0]
        self.ax3d.plot(xs, ys, zs, "w-", lw=2)

        arm_len = self.rotor_radius * 1.1
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            self.ax3d.plot([0, sx * arm_len], [0, sy * arm_len], [0, 0], "w-", lw=1.5)
            theta = np.linspace(0, 2*np.pi, 40)
            x_c = sx * arm_len
            y_c = sy * arm_len
            r = self.rotor_radius
            self.ax3d.plot(x_c + r * np.cos(theta),
                           y_c + r * np.sin(theta),
                           np.zeros_like(theta),
                           color="cyan", lw=1.5)

    def run_comparison(self, event):
        """
        Offline comparison: step vs slow vs fast thrust profiles
        with same rotor and ν. Plots on ax_graph.
        """
        modes = ["step", "slow", "fast"]
        colors = ["orange", "deepskyblue", "red"]
        results = {}

        for mode in modes:
            t = 0.0
            rings = []
            time_hist = []
            gamma_hist = []
            last_emit = -999.0
            dt = 0.02
            while t < 2.0:
                t += dt
                # emission
                if t - last_emit > 0.12:
                    T_now = thrust_profile(t, mode, self.T_max)
                    if T_now > 0.1:
                        Gamma = thrust_to_circulation(T_now, self.rotor_radius, RHO_AIR)
                        nu_eff = NU_AIR * self.visc_scale
                        rings.append(VortexRingSimple(self.rotor_radius, -0.02, Gamma, nu_eff))
                        last_emit = t
                # update
                for r in rings:
                    r.update(dt)
                if len(rings) > 25:
                    rings.pop(0)
                total_gamma = sum(max(r.Gamma, 0.0) for r in rings)
                time_hist.append(t)
                gamma_hist.append(total_gamma)
            results[mode] = (np.array(time_hist), np.array(gamma_hist), colors[modes.index(mode)])

        self.compare_results = results
        print("Comparison profile data ready – see overlay in graph.")

    def update(self, frame):
        if self.running:
            self.step_sim()

        # 3D view
        self.ax3d.cla()
        self.ax3d.set_title("Drone and vortex rings")
        self.ax3d.set_xlabel("x [m]")
        self.ax3d.set_ylabel("y [m]")
        self.ax3d.set_zlabel("z [m]")
        self.ax3d.set_xlim(-0.35, 0.35)
        self.ax3d.set_ylim(-0.35, 0.35)
        self.ax3d.set_zlim(-0.7, 0.2)
        self.ax3d.view_init(elev=25, azim=-60)

        self.draw_drone()

        for r in self.rings:
            theta = np.linspace(0, 2*np.pi, 80)
            x = r.R * np.cos(theta)
            y = r.R * np.sin(theta)
            z = np.full_like(theta, r.z)
            # color intensity based on Γ
            gamma_norm = min(abs(r.Gamma) / 5.0, 1.0)
            color = (1.0, 0.5 + 0.5*(1-gamma_norm), 0.0)
            self.ax3d.plot(x, y, z, color=color, lw=2 + 30*r.a)

        # Info + dimensionless panel
        self.ax_info.cla()
        self.ax_info.axis("off")
        Re_gamma, St, regime = self.compute_dimensionless()
        text_lines = [
            "Physical picture:",
            "",
            "- White shape: small quadcopter drone.",
            "- Orange rings: vortex rings (circulation Γ [m²/s]) in the wake.",
            "- Their speed comes from Γ, radius R, and core size a.",
            "",
            f"Mode: {self.mode}",
            "",
            "Key dimensionless numbers (order of magnitude):",
        ]
        if not np.isnan(Re_gamma):
            text_lines.append(f"  Re_Γ ≈ Γ / ν ≈ {Re_gamma:6.2e} [-]")
            text_lines.append(f"  St  ≈ f R / U ≈ {St:5.2f} [-]")
            text_lines.append(f"  Wake regime: {regime}")
        else:
            text_lines.append("  (Waiting for rings to form...)")
        text_lines += [
            "",
            "Controls (SI units):",
            "  Max thrust [N]     : peak rotor thrust",
            "  Rotor radius [m]   : rotor size",
            "  ν scale [-]        : viscosity multiplier",
            "  Time scale [-]     : animation speed",
            "",
            "Buttons:",
            "  Pause/Start : stop or continue simulation",
            "  Reset       : clear wake and restart",
            "  Save CSV    : export t vs ΣΓ(t) in [m²/s]",
            "  Compare     : overlay Γ(t) for step/slow/fast",
        ]
        self.ax_info.text(0.0, 1.0, "\n".join(text_lines),
                          va="top", fontsize=9, color="w")

        # Graph: ΣΓ(t) + optional comparison overlay
        self.ax_graph.cla()
        self.ax_graph.set_title("Total circulation ΣΓ(t) in wake")
        self.ax_graph.set_xlabel("time [s]")
        self.ax_graph.set_ylabel("ΣΓ [m²/s]")
        if len(self.time_hist) > 1:
            self.ax_graph.plot(self.time_hist, self.gamma_hist,
                               color="lime", lw=2, label=f"{self.mode} (current)")
        if self.compare_results is not None:
            for mode, (t_arr, g_arr, col) in self.compare_results.items():
                self.ax_graph.plot(t_arr, g_arr, color=col, lw=1.5,
                                   linestyle="--", label=f"{mode} (compare)")
        if self.ax_graph.lines:
            self.ax_graph.grid(True, alpha=0.3)
            self.ax_graph.legend(loc="upper left", fontsize=8)

        return []


if __name__ == "__main__":
    app = DroneVortexApp()
    plt.show()

