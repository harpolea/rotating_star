import numpy as np
from scipy.special import lpmn, factorial
from scipy.integrate import quad

import eos


class Star(object):

    def __init__(self, rotation_law, _eos, mesh_size, rmax=1.05, G=1):
        """
        Constructor
        """
        laws = {"rigid": self.rigid_rotation, "v-constant": self.v_constant_rotation,
                "j-constant": self.j_constant_rotation}
        self.rotation_law = laws[rotation_law]

        eoses = {"polytrope": eos.Polytrope, "wd": eos.WD_matter}

        self.eos = eoses[_eos]

        self.G = G

        self.mesh_size = mesh_size
        self.rmax = rmax

        # initialize mesh
        self.rho = np.zeros(self.mesh_size)
        self.Phi = np.zeros(self.mesh_size)

        # initialize coords

        self.phi_coords = np.pi * \
            np.array(range(self.mesh_size[0])) / (self.mesh_size[0] - 1)

        self.mu_coords = np.array(
            range(self.mesh_size[1])) / (self.mesh_size[1] - 1)

        self.r_coords = rmax * \
            np.array(range(self.mesh_size[2])) / (self.mesh_size[2] - 1)

        self.Psi = np.zeros(self.mesh_size)
        self.Psi[:, :, :] = -0.5 * self.r_coords[np.newaxis, np.newaxis,
                                                 :]**2 * (1 - self.mu_coords[np.newaxis, :, np.newaxis]**2)

        self.H = np.zeros(self.mesh_size)
        self.Omega2 = np.zeros(self.mesh_size)
        self.C = np.zeros(self.mesh_size)

    @staticmethod
    def rigid_rotation(r, omegac, d):
        omega = omegac
        Cv = omegac**2
        chi = -0.5 * r**2

        return omega, Cv, chi

    @staticmethod
    def v_constant_rotation(r, omegac, d):
        omega = omegac / (1 + r / d)
        Cv = omegac**2
        chi = -d**2 * (d / (d + r) + np.log(d + r))

        return omega, Cv, chi

    @staticmethod
    def j_constant_rotation(r, omegac, d):
        omega = omegac / (1 + r**2 / d**2)
        Cv = omegac**2
        chi = 0.5 * d**4 / (d**2 + r**2)

        return omega, Cv, chi

    def SCF_step(self):
        """ Implements self-consistent field algorithm """
        lmax = 32

        M, K, N = self.mesh_size
        ph = self.phi_coords
        mu = self.mu_coords
        r = self.r_coords

        def D1(t, s, m):
            sum = 0

            for u in range(0, M - 2, 2):
                sum += 1 / 6 * (ph[u + 2] - ph[u]) * (np.cos(m * ph[u] * self.rho[u, t, s])
                                                      + 4 * np.cos(m * ph[u + 1] * self.rho[u + 1, t, s]) +
                                                      np.cos(m * ph[u + 2] * self.rho[u + 2, t, s]))

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
                    self.Phi[m, k, n] = calc_Phi(m, k, n)

        # update the enthalpy

        Omega2 = self.eos.Omega2(self.Phi, self.Psi)
        C = self.eos.C(self.Phi, self.Psi)

        H = C - self.Phi - Omega2 * self.Psi

        # use new enthalpy and Phi to calculate the density

        self.rho = self.eos.rho_from_h(H)

        # calculate the errors

        H_err = np.max(H - self.H) / np.max(H)
        Omega2_err = np.max(Omega2 - self.Omega2) / np.max(Omega2)
        C_err = np.max(C - self.C) / np.max(self.C)

        # set variables to new values

        self.H = H
        self.Omega2 = Omega2
        self.C = C

        return H_err, Omega2_err, C_err

    def SCF(self):
        delta = 1e-3
        max_steps = 100

        H_err = 1
        Omega2_err = 1
        C_err = 1

        counter = 0

        while H_err > delta and Omega2_err > delta and C_err > delta and counter < max_steps:
            H_err, Omega2_err, C_err = self.SCF_step()
            counter += 1
