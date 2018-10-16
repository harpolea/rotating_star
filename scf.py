import numpy as np
from scipy.special import lpmn, factorial
from scipy.integrate import quad

class SCF(object):

    def __init__(self, star):
        star.star = star

    def step(self):
        """ Implements self-consistent field algorithm """
        lmax = 32

        star = self.star

        M, K, N = star.mesh_size
        ph = star.phi_coords
        mu = star.mu_coords
        r = star.r_coords

        def D1(t, s, m):
            sum = 0

            for u in range(0, M - 2, 2):
                sum += 1 / 6 * (ph[u + 2] - ph[u]) * (np.cos(m * ph[u] * star.rho[u, t, s])
                                                      + 4 * np.cos(m * ph[u + 1] * star.rho[u + 1, t, s]) +
                                                      np.cos(m * ph[u + 2] * star.rho[u + 2, t, s]))

            return 2 * sum

        def D2(s, l, m):
            sum = 0

            for t in range(0, K - 2, 2):
                sum += (1 / 6) * (mu[t + 2] - mu[t]) * (lpmn(l, m, mu[t]) * D1(t, s, m) +
                                                        4 * lpmn(l, m, mu[t+1]) * D1(t + 1, s, m) +
                                                        lpmn(l, m, mu[t + 2]) * D1(t + 2, s, m))

            return 2 * sum

        def D3(l, m, k):
            sum = 0

            def fl(r_dash, r):
                if r_dash < r:
                    return r_dash**(l + 2) / r**(l + 1)
                else:
                    return r**l / r_dash**(l - 1)

            for s in range(0, N - 2, 2):
                sum += (1 / 6) * (r[s + 2] - r[s]) * (fl(r[s], r[k]) * D2(s, l, m) +
                                                      4 * fl(r[s + 1], r[k]) * D2(s + 1, l, m) +
                                                      fl(r[s + 2], r[k]) * D2(s + 2, l, m))

            return sum

        def calc_Phi(i, j, k):
            Phi = 0

            for l in range(lmax):
                for m in range(l):
                    if m == 0:
                        eps = 1
                    else:
                        eps = 2
                    Phi += eps * factorial(1 - m) / factorial(1 + m) * \
                        D3(l, m, k) * lpmn(l, m, mu[j]) * np.cos(m * ph[i])

            return Phi

        # calculate Phi across grid
        for n in range(N):
            for k in range(K):
                for m in range(M):
                    star.Phi[m, k, n] = calc_Phi(m, k, n)

        # update the enthalpy

        Omega2 = star.eos.Omega2(star.Phi, star.Psi)
        C = star.eos.C(star.Phi, star.Psi)

        H = C - star.Phi - Omega2 * star.Psi

        # use new enthalpy and Phi to calculate the density

        star.rho = star.eos.rho_from_h(H)

        # calculate the errors

        H_err = np.max(H - star.H) / np.max(H)
        Omega2_err = np.max(Omega2 - star.Omega2) / np.max(Omega2)
        C_err = np.max(C - star.C) / np.max(star.C)

        # set variables to new values

        star.H = H
        star.Omega2 = Omega2
        star.C = C

        return H_err, Omega2_err, C_err

    def solve(self):
        delta = 1e-3
        max_steps = 100

        H_err = 1
        Omega2_err = 1
        C_err = 1

        counter = 0

        while H_err > delta and Omega2_err > delta and C_err > delta and counter < max_steps:
            H_err, Omega2_err, C_err = self.SCF_step()
            counter += 1
