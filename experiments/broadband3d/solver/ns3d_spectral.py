"""3D incompressible Navier-Stokes, pseudo-spectral, periodic box [0,2pi)^3.

Rotational form:  du/dt = u x omega - grad(P) + nu*lap(u),  div u = 0,
with omega = curl u.  Nonlinear term u x omega is computed in physical space and
projected divergence-free in Fourier space; 2/3-rule dealiasing; explicit RK4 for the
full RHS (nonlinear + viscous).  Conserved diagnostics: energy E and helicity H.

Helicity is THE quantity that bridges to the graph/knot picture: for a collection of
vortex tubes of circulation Gamma, H = sum Gamma_i Gamma_j Lk_ij + self-twist terms
(Moffatt 1969).  This module is validated in __main__ against two exact facts:
  - a Beltrami (ABC) flow has omega = u, hence H = integral u.u = 2E  (ratio 1),
  - the Taylor-Green vortex has H = 0 and energy that decays monotonically.

fp64. No RNG.
"""
import numpy as np


class NS3D:
    def __init__(self, N=64, nu=1e-2, L=2 * np.pi):
        self.N, self.nu, self.L = N, nu, L
        k1 = np.fft.fftfreq(N, d=L / N) * 2 * np.pi      # angular wavenumbers
        self.kx, self.ky, self.kz = np.meshgrid(k1, k1, k1, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_nz = self.k2.copy()
        self.k2_nz[0, 0, 0] = 1.0                         # avoid /0 for projection
        # 2/3 dealiasing mask
        kmax = np.max(np.abs(k1)) * 2.0 / 3.0
        self.dealias = ((np.abs(self.kx) <= kmax) &
                        (np.abs(self.ky) <= kmax) &
                        (np.abs(self.kz) <= kmax)).astype(float)

    # ---- FFT helpers ----
    def fft(self, u):    return np.fft.fftn(u, axes=(1, 2, 3))
    def ifft(self, uh):  return np.real(np.fft.ifftn(uh, axes=(1, 2, 3)))

    def project(self, uh):
        """Leray projection onto divergence-free fields."""
        kdotu = self.kx * uh[0] + self.ky * uh[1] + self.kz * uh[2]
        uh = uh.copy()
        uh[0] -= self.kx * kdotu / self.k2_nz
        uh[1] -= self.ky * kdotu / self.k2_nz
        uh[2] -= self.kz * kdotu / self.k2_nz
        return uh

    def curl_h(self, uh):
        wx = 1j * (self.ky * uh[2] - self.kz * uh[1])
        wy = 1j * (self.kz * uh[0] - self.kx * uh[2])
        wz = 1j * (self.kx * uh[1] - self.ky * uh[0])
        return np.array([wx, wy, wz])

    def rhs(self, uh):
        """du/dt in Fourier space (rotational form, projected, dealiased)."""
        u = self.ifft(uh)
        wh = self.curl_h(uh)
        w = self.ifft(wh)
        # u x omega
        cx = u[1] * w[2] - u[2] * w[1]
        cy = u[2] * w[0] - u[0] * w[2]
        cz = u[0] * w[1] - u[1] * w[0]
        ch = self.fft(np.array([cx, cy, cz])) * self.dealias
        ch = self.project(ch)
        return ch - self.nu * self.k2 * uh

    def step_rk4(self, uh, dt):
        k1 = self.rhs(uh)
        k2 = self.rhs(uh + 0.5 * dt * k1)
        k3 = self.rhs(uh + 0.5 * dt * k2)
        k4 = self.rhs(uh + dt * k3)
        return uh + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # ---- diagnostics (physical space, exact quadrature on the periodic grid) ----
    def energy(self, uh):
        u = self.ifft(uh)
        return 0.5 * np.mean(u[0]**2 + u[1]**2 + u[2]**2) * self.L**3

    def helicity(self, uh):
        u = self.ifft(uh)
        w = self.ifft(self.curl_h(uh))
        return np.mean(u[0]*w[0] + u[1]*w[1] + u[2]*w[2]) * self.L**3

    def enstrophy(self, uh):
        w = self.ifft(self.curl_h(uh))
        return 0.5 * np.mean(w[0]**2 + w[1]**2 + w[2]**2) * self.L**3

    def max_divergence(self, uh):
        divh = 1j * (self.kx*uh[0] + self.ky*uh[1] + self.kz*uh[2])
        div = np.real(np.fft.ifftn(divh))
        return np.max(np.abs(div))


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def abc_flow(N, A=1.0, B=1.0, C=1.0, L=2*np.pi):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.array([A*np.sin(Z) + C*np.cos(Y),
                  B*np.sin(X) + A*np.cos(Z),
                  C*np.sin(Y) + B*np.cos(X)])
    return u


def taylor_green(N, L=2*np.pi):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.array([ np.cos(X)*np.sin(Y)*np.sin(Z),
                  -np.sin(X)*np.cos(Y)*np.sin(Z),
                   np.zeros_like(X)])
    return u


def main():
    import json
    out = {"meta": {"fp64": True}}
    print("=" * 68)
    print("VALIDATION 1: Beltrami (ABC) flow  =>  H = 2E exactly")
    print("=" * 68)
    N = 64
    sim = NS3D(N=N, nu=0.0)
    uh = sim.fft(abc_flow(N))
    uh = sim.project(uh)
    E, H = sim.energy(uh), sim.helicity(uh)
    print(f"  N={N}  E={E:.6f}  H={H:.6f}  H/2E={H/(2*E):.8f}  (expect 1)")
    print(f"  max|div u| = {sim.max_divergence(uh):.2e}")
    out["beltrami"] = {"E": float(E), "H": float(H), "H_over_2E": float(H/(2*E)),
                       "max_div": float(sim.max_divergence(uh))}

    print("\n" + "=" * 68)
    print("VALIDATION 2: Taylor-Green  =>  H=0, energy decays monotonically")
    print("=" * 68)
    sim = NS3D(N=N, nu=1e-2)
    uh = sim.project(sim.fft(taylor_green(N)))
    dt = 0.01
    E0 = sim.energy(uh)
    print(f"  t=0.00  E={E0:.6f}  H={sim.helicity(uh):+.2e}  ens={sim.enstrophy(uh):.4f}")
    traj = [{"t": 0.0, "E": float(E0), "H": float(sim.helicity(uh)),
             "ens": float(sim.enstrophy(uh))}]
    Eprev = E0
    mono = True
    for n in range(1, 201):
        uh = sim.step_rk4(uh, dt)
        if n % 40 == 0:
            E, H, ens = sim.energy(uh), sim.helicity(uh), sim.enstrophy(uh)
            print(f"  t={n*dt:.2f}  E={E:.6f}  H={H:+.2e}  ens={ens:.4f}  "
                  f"div={sim.max_divergence(uh):.1e}")
            traj.append({"t": n*dt, "E": float(E), "H": float(H), "ens": float(ens)})
        E = sim.energy(uh)
        if E > Eprev + 1e-10:
            mono = False
        Eprev = E
    print(f"  energy monotonically decreasing: {mono}")
    print(f"  |H| stayed ~0: max={max(abs(p['H']) for p in traj):.2e}")
    out["taylor_green"] = {"trajectory": traj, "energy_monotone_decreasing": mono,
                           "max_abs_H": max(abs(p['H']) for p in traj)}

    with open("ns3d_validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  -> ns3d_validation_results.json")


if __name__ == "__main__":
    main()
