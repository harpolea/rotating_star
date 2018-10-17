from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import lpmv, factorial
from scipy.integrate import quad


class Solver(metaclass=ABCMeta):

    def __init__(self, star):
        self.star = star

    @abstractmethod
    def solve(self, max_steps=100):
        pass


class SCF(Solver):

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
                sum += (1 / 6) * (mu[t + 2] - mu[t]) * (lpmv(m, l, mu[t]) * D1(t, s, m) +
                                                        4 * lpmv(m, l, mu[t + 1]) * D1(t + 1, s, m) +
                                                        lpmv(m, l, mu[t + 2]) * D1(t + 2, s, m))

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
                    Phi -= eps * factorial(1 - m) / factorial(1 + m) * \
                        D3(l, m, k) * lpmv(m, l, mu[j]) * np.cos(m * ph[i])

            return Phi

        # calculate Phi across grid
        for n in range(N):
            for k in range(K):
                for m in range(M):
                    star.Phi[m, k, n] = calc_Phi(m, k, n)

        print(f'Phi = {star.Phi[0,0,:]}')

        # update the enthalpy

        Omega2 = star.eos.Omega2(star.eos, star.Phi, star.Psi)
        C = star.eos.C(star.eos, star.Phi, star.Psi)

        H = C - star.Phi - Omega2 * star.Psi

        # use new enthalpy and Phi to calculate the density

        star.rho = star.eos.rho_from_h(star.eos, H)

        print(f"rho = {star.rho[0,0,:]}")

        # calculate the errors

        H_err = np.max(np.abs(H - star.H)) / np.max(np.abs(H))

        if np.max(Omega2) == 0:
            if np.abs(Omega2 - star.Omega2) == 0:
                Omega2_err = 0
            else:
                Omega2_err = 1
        else:
            Omega2_err = np.abs(Omega2 - star.Omega2) / np.abs(Omega2)

        if np.max(star.C) == 0:
            if np.abs(C - star.C) == 0:
                C_err = 0
            else:
                C_err = 1
        else:
            C_err = np.abs(C - star.C) / np.abs(star.C)

        # set variables to new values

        star.H = H
        star.Omega2 = Omega2
        star.C = C

        print(
            f"Errors: H_err = {H_err}, Omega2_err = {Omega2_err}, C_err = {C_err}")

        return H_err, Omega2_err, C_err

    def solve(self, max_steps=100):
        """
        Iterate single step until errors in enthalpy, spin rate and C are small
        """
        delta = 1e-3

        H_err = 1
        Omega2_err = 1
        C_err = 1

        for i in range(max_steps):
            print(f"Step {i}")
            H_err, Omega2_err, C_err = self.step()

            if H_err < delta and Omega2_err < delta and C_err < delta:
                print("Solution found!")
                break

    def calc_mass(self):
        """ Integrate over star to get the total mass """

        star = self.star

        M, K, N = star.mesh_size
        ph = star.phi_coords
        mu = star.mu_coords
        r = star.r_coords

        def Q1(j, k):
            sum = 0

            for i in range(0, M - 2, 2):
                sum += (1 / 6) * (ph[i + 2] - ph[i]) * (star.rho[i, j, k] +
                                                        4 * star.rho[i + 1, j, k] +
                                                        star.rho[i + 2, j, k])

            return 2 * sum

        def Q2(k):
            sum = 0

            for j in range(0, K - 2, 2):
                sum += (1 / 6) * (mu[j + 2] - mu[j]) * \
                    (Q1(j, k) + 4 * Q1(j + 1, k) + Q1(j + 2, k))

            return 2 * sum

        mass = 0

        for k in range(0, N - 2, 2):
            mass += (1 / 6) * (r[k + 2] - r[k]) * (r[k]**2 * Q2(k) +
                                                   4 * r[k + 1]**2 * Q2(k + 1) +
                                                   r[k + 2]**2 * Q2(k + 2))

        return mass

    def calc_gravitational_energy(self):
        """ Integrate over star to get the total gravitational energy """

        star = self.star

        M, K, N = star.mesh_size
        ph = star.phi_coords
        mu = star.mu_coords
        r = star.r_coords

        def S1(j, k):
            sum = 0

            for i in range(0, M - 2, 2):
                sum += (1 / 6) * (ph[i + 2] - ph[i]) * (star.rho[i, j, k] * star.Phi[i, j, k] + 4 *
                                                        star.rho[i + 1, j, k] * star.Phi[i + 1, j, k] +
                                                        star.rho[i + 2, j, k] * star.Phi[i + 2, j, k])
            return 2 * sum

        def S2(k):
            sum = 0

            for j in range(0, K - 2, 2):
                sum += (1 / 6) * (mu[j + 2] - mu[j]) * \
                    (S1(j, k) + 4 * S1(j + 1, k) + S1(j + 2, k))

            return 2 * sum

        W = 0

        for k in range(0, N - 2, 2):
            W -= 0.5 * (1 / 6) * (r[k + 2] - r[k]) * (r[k]**2 * S2(k) +
                                                      4 * r[k + 1]**2 * S2(k + 1) +
                                                      r[k + 2]**2 * S2(k + 2))

        return W
