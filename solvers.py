from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import lpmv, factorial, eval_legendre
from scipy.optimize import fsolve


class Solver(metaclass=ABCMeta):

    def __init__(self, star):
        self.star = star

    @abstractmethod
    def initialize_solver(self, parameters):
        pass

    @abstractmethod
    def solve(self, max_steps=100):
        pass

    @abstractmethod
    def calc_mass(self):
        pass

    @abstractmethod
    def calc_gravitational_energy(self):
        pass


class SCF(Solver):

    def initialize_solver(self, parameters):
        pass

    def step(self):
        """ Implements 2d self-consistent field algorithm """
        lmax = 32

        star = self.star

        K, N = star.mesh_size
        mu = star.mu_coords
        r = star.r_coords

        def D1(k, n):
            return 1 / 6 * np.sum((mu[2::2] - mu[:-2:2]) *
                                  (eval_legendre(2 * n, mu[:-2:2]) * star.rho[:-2:2, k] +
                                   4 * eval_legendre(2 * n, mu[1:-1:2]) * star.rho[1:-1:2, k] +
                                   eval_legendre(2 * n, mu[2::2]) * star.rho[2::2, k]))

        def D2(n, j):
            sum = 0

            def fl(r_dash, r, l=2 * n):
                if r_dash < r:
                    return r_dash**(l + 2) / r**(l + 1)
                else:
                    return r**l / r_dash**(l - 1)

            for k in range(0, N - 2, 2):
                sum += (r[k + 2] - r[k]) * (fl(r[k], r[j]) * D1(k, n) +
                                            4 * fl(r[k + 1], r[j]) * D1(k + 1, n) +
                                            fl(r[k + 2], r[j]) * D1(k + 2, n))

            return sum / 6

        def calc_Phi(i, j):
            Phi = 0

            for n in range(lmax + 1):
                Phi -= 4 * np.pi * D2(n, j) * eval_legendre(2 * n, mu[i])

            return Phi

        # calculate Phi across grid
        for n in range(N):
            for k in range(K):
                star.Phi[k, n] = calc_Phi(k, n)

        # print(f'Phi = {star.Phi[0,:]}')

        # update the enthalpy

        Omega2 = star.eos.Omega2(star.Phi, star.Psi)
        C = star.eos.C(star.Phi, star.Psi)

        H = C - star.Phi - Omega2 * star.Psi

        # use new enthalpy and Phi to calculate the density

        star.rho = star.eos.rho_from_h(H)
        star.rho /= np.max(star.rho)

        # print(f"rho = {np.average(star.rho, axis=0)}")

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
        delta = 1e-4

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

        K, N = star.mesh_size
        mu = star.mu_coords
        r = star.r_coords

        def Q1(j):
            return np.sum((mu[2::2] - mu[:-2:2]) *
                          (star.rho[:-2:2, j] +
                           4 * star.rho[1:-1:2, j] +
                           star.rho[2::2, j]))

        mass = 0

        for j in range(0, N - 2, 2):
            mass += (r[j + 2] - r[j]) * (r[j]**2 * Q1(j) +
                                         4 * r[j + 1]**2 * Q1(j + 1) +
                                         r[j + 2]**2 * Q1(j + 2))

        return 2 / 3 * np.pi * mass

    def calc_gravitational_energy(self):
        """ Integrate over star to get the total gravitational energy """

        star = self.star

        K, N = star.mesh_size
        mu = star.mu_coords
        r = star.r_coords

        def S1(j):
            return np.sum((mu[2::2] - mu[:-2:2]) * (star.rho[:-2:2, j] * star.Phi[:-2:2, j] +
                                                    4 * star.rho[1:-1:2, j] * star.Phi[1:-1:2, j] +
                                                    star.rho[2::2, j] * star.Phi[2::2, j]))

        W = 0

        for j in range(0, N - 2, 2):
            W += (r[j + 2] - r[j]) * (r[j]**2 * S1(j) +
                                      4 * r[j + 1]**2 * S1(j + 1) +
                                      r[j + 2]**2 * S1(j + 2))

        return -1 / 3 * np.pi * W


class Newton(Solver):
    """ Implements the Newton solver of Aksenov and Blinnikov """

    def initialize_solver(self, parameters):

        self.Chi = self.star.rotation_law.Chi(self.star.omegabar)
        Phi = self.star.Phi
        r2d = np.zeros(self.star.mesh_size)
        r2d[:, :] = self.star.r_coords[np.newaxis, :]

        Phi[r2d < 1] = -1.5
        Phi[r2d == 1] = -1
        Phi[r2d > 1] = -0.5

        H = self.H_from_Phi(Phi)

        self.star.H = H  # self.star.eos.h_from_rho(self.star.rho)

        self.star.rho = self.star.eos.rho_from_h(H)

        print(f"rho = {self.star.rho[0,:]}")
        print(f"H = {H[0,:]}")

    def H_from_Phi(self, Phi):
        A = self.star.eos.A
        B = self.star.eos.B
        Chi = self.Chi

        C_Psi = (Phi[B] - Phi[A]) / (Chi[A] - Chi[B])

        Psi = C_Psi * Chi

        C = 0.5 * (Phi[A] + Phi[B] + Psi[A] + Psi[B])

        return C - Phi - Psi

    def step(self):

        Phi = self.star.Phi
        Phi0 = Phi[0, 0]

        Chi = self.Chi
        Chi0 = Chi[0, 0]
        H = self.H_from_Phi(Phi)
        H0 = H[0, 0]

        rho = self.star.eos.rho_from_h(H)

        A = self.star.eos.A
        B = self.star.eos.B

        C_Psi = (Phi[B] - Phi[A]) / (Chi[A] - Chi[B])

        Psi = C_Psi * Chi

        C = 0.5 * (Phi[A] + Phi[B] + Psi[A] + Psi[B])

        _B = C - Phi - Psi
        B0 = _B[0, 0]

        print(f"B0 = {B0}, Phi0 = {Phi0}, Psi0 = {Psi[0,0]}, C = {C}")
        # print(f"B = {_B}")

        rho_H_dash = self.star.eos.rho_H_dash(H)

        rho0 = rho[0, 0]

        print(f"rho0 = {rho0}")

        gPhi = 4 * np.pi * self.star.G * rho_H_dash * H0 / (rho0 * B0)

        gA = - gPhi * ((Chi[A] - Chi) / (Chi[A] - Chi[B]) -
                       _B / B0 * (Chi[A] - Chi0) / (Chi[A] - Chi[B]))

        gB = - gPhi * ((Chi - Chi[B]) / (Chi[A] - Chi[B]) -
                       _B / B0 * (Chi0 - Chi[B]) / (Chi[A] - Chi[B]))

        g0 = - gPhi * _B / B0

        R = (4 * np.pi * self.star.G * rho - gA *
             Phi[A] - gB * Phi[B] - g0 * Phi0 - gPhi * Phi).flatten()

        r = self.star.r_coords
        th = self.star.theta_coords

        r2d = np.zeros(self.star.mesh_size)
        r2d[:, :] = r[np.newaxis, :]

        th2d = np.zeros(self.star.mesh_size)
        th2d[:, :] = th[:, np.newaxis]

        dr = r[1] - r[0]
        h = 1 / (self.star.mesh_size[1] - A[1])
        dth = th[1] - th[0]

        # find where r >=1
        indx = 0
        for i, rr in enumerate(r):
            if rr >= 1:
                indx = i
                break

        # def laplacian(phi):
        #     _laplacian = np.zeros_like(phi[:, 1:-1])
        #
        #     # r part
        #     _laplacian[:, :] = 1 / (dr**2 * r2d[:, 1:-1]) * 0.5 * \
        #         ((r2d[:, 2:]**2 + r2d[:, 1:-1]**2) * (Phi[:, 2:] - Phi[:, 1:-1]) -
        #          (r2d[:, 1:-1]**2 + r2d[:, :-2]**2) * (Phi[:, 1:-1] - Phi[:, :-2]))
        #
        #     _laplacian[:, indx:] = 1 / (r2d[:, indx:-1]**4 * h**2) * (
        #         Phi[:, indx - 1:-2] - 2 * Phi[:, indx:-1] + Phi[:, indx + 1:])
        #
        #     # theta part
        #     _laplacian[1:-1, :] += 1 / (r2d[1:-1, 1:-1]**2 * np.sin(th2d[1:-1, 1:-1]) * dth**2) * 0.5 * \
        #         ((np.sin(th2d[2:, 1:-1]) + np.sin(th2d[1:-1, 1:-1]) * (Phi[2:, 1:-1] - Phi[1:-1, 1:-1]) -
        #           (np.sin(th2d[1:-1, 1:-1]) + np.sin(th2d[:-2, 1:-1])) * (Phi[1:-1, 1:-1] - Phi[:-2, 1:-1])))
        #
        #     # boundaries
        #     _laplacian[0, :] += 4 / r2d[0, 1:-1]**2 * \
        #         (Phi[1, 1:-1] - Phi[0, 1:-1]) / dth**2
        #
        #     _laplacian[-1, :] += 2 / r2d[-1, 1:-1]**2 * \
        #         (Phi[-2, 1:-1] - Phi[-1, 1:-1]) / dth**2
        #
        #     # _laplacian[1:-1,1:-1] =  (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dr**2 + \
        #     #     1 / self.star.r_coords[1:-1] * (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dr) + \
        #     #     1 / self.star.r_coords[1:-1]**2 * ((phi[2:, 1:-1] - phi[1:-1, 1:-1]) / (self.star.theta_coords[2:] - self.star.theta_coords[1:-1]) +
        #     #                                        (phi[:-2, 1:-1] - phi[1:-1, 1:-1]) / (self.star.theta_coords[1:-1] - self.star.theta_coords[:-2]))
        #     return _laplacian

        # calculate LHS matrix operator

        nx = self.star.mesh_size[0] * self.star.mesh_size[1]

        M = np.zeros((nx, nx))

        # M[:, :-2] += 1/(dr**2 * r[1:-1]**2) * 0.5 * (r[1:-1]**2 + r[:-2]**2)

        for j in range(self.star.mesh_size[0]):
            for i in range(1, self.star.mesh_size[1]-1):
                ix = j * self.star.mesh_size[1] + i
                M[ix, ix-1] += 1/(dr**2 * r[i]**2) * 0.5 * (r[i]**2 + r[i-1]**2)
                M[ix, ix] -= 1/(dr**2 * r[i]**2) * 0.5 * (2*r[i]**2 + r[i-1]**2 + r[i+1]**2)
                M[ix, ix+1] += 1/(dr**2 * r[i]**2) * 0.5 * (r[i]**2 + r[i+1]**2)

        for j in range(1, self.star.mesh_size[0]-1):
            for i in range(1,self.star.mesh_size[1]):

                jm = (j-1) * self.star.mesh_size[1] + i
                jp = (j+1) * self.star.mesh_size[1] + i
                M[ix, jm] += 1/(r[i]**2 * np.sin(th[j]) * dth**2) * 0.5 * (np.sin(th[j]) + np.sin(th[j-1]))
                M[ix, ix] -= 1/(r[i]**2 * np.sin(th[j]) * dth**2) * 0.5 * (2 * np.sin(th[j]) + np.sin(th[j-1]) + np.sin(th[j+1]))
                M[ix, jp] +=  1/(r[i]**2 * np.sin(th[j]) * dth**2) * 0.5 * (np.sin(th[j]) + np.sin(th[j+1]))

        # do boundaries
        for i in range(1,self.star.mesh_size[1]):
            ix = i
            jp = self.star.mesh_size[1] + i
            M[ix, ix] -= 4/(r[i]**2 * dth**2)
            M[ix, jp] += 4/(r[i]**2 * dth**2)

            ix = (self.star.mesh_size[0]-1)*self.star.mesh_size[1] + i
            jm = (self.star.mesh_size[0]-2)*self.star.mesh_size[1] + i
            M[ix,jm] += 2/(r[i]**2 * dth**2)
            M[ix,ix] -= 2/(r[i]**2 * dth**2)

        # print(M)

        # Phi = fsolve(root_solve, self.star.Phi.flatten()
        #              ).reshape(self.star.mesh_size)

        # add on g's
        M[:,:] += np.diag(gPhi.flatten())

        A_idx = A[0] * self.star.mesh_size[1] + A[1]
        B_idx = B[0] * self.star.mesh_size[1] + B[1]

        M[:,A_idx] += gA.flatten()
        M[:, B_idx] += gB.flatten()

        for j in range(self.star.mesh_size[0]):
            ix = j * self.star.mesh_size[1]

            M[:,ix] += g0.flatten()

        Phi = np.linalg.solve(M, R).reshape(self.star.mesh_size)


        print(f"Phi = {Phi[0,:]}")

        H = self.H_from_Phi(Phi)

        # self.star.rho[1:-1,1:-1] = laplacian(self.star.Phi)[1:-1,1:-1] / (4 * np.pi * self.star.G)

        self.star.rho = self.star.eos.rho_from_h(H)

        print(f"rho = {self.star.rho[0,:]}")

        # print(f"H = {self.star.H}")

        # calculate the errors

        # Omega2 = self.star.eos.Omega2(Phi, self.star.Psi)
        # C = self.star.eos.C(self.star.eos, Phi, self.star.Psi)

        # H = self.star.eos.h_from_rho(self.star.eos, self.star.rho)

        H_err = np.max(np.abs(H - self.star.H)) / np.max(np.abs(H))

        Phi_err = np.max(np.abs(Phi - self.star.Phi)) / np.max(np.abs(Phi))

        # set variables to new values

        self.star.H = H
        self.star.Phi = Phi

        print(
            f"Errors: H_err = {H_err}, Phi_err = {Phi_err}")

        return H_err, Phi_err, 0

    def solve(self, max_steps=100):
        delta = 1e-4

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
        pass

    def calc_gravitational_energy(self):
        pass


class SCF3(Solver):

    def initialize_solver(self, parameters):
        pass

    def step(self):
        """ Implements 3d self-consistent field algorithm. This does not currently work. """
        lmax = 32

        star = self.star

        M, K, N = star.mesh_size
        ph = star.phi_coords
        mu = star.mu_coords
        r = star.r_coords

        def D1(t, s, m):
            return 1 / 3 * np.sum((ph[2::2] - ph[:-2:2]) *
                                  (np.cos(m * ph[:-2:2] * star.rho[:-2:2, t, s])
                                   + 4 * np.cos(m * ph[1:-1:2] * star.rho[1:-1:2, t, s]) +
                                   np.cos(m * ph[2::2] * star.rho[2::2, t, s])))

        def D2(s, l, m):
            sum = 0
            for t in range(0, K - 2, 2):
                sum += (mu[t + 2] - mu[t]) * (lpmv(m, l, mu[t]) * D1(t, s, m) +
                                              4 * lpmv(m, l, mu[t + 1]) * D1(t + 1, s, m) +
                                              lpmv(m, l, mu[t + 2]) * D1(t + 2, s, m))

            return sum / 3

        def D3(l, m, k):
            sum = 0

            def fl(r_dash, r):
                if r_dash < r:
                    return r_dash**(l + 2) / r**(l + 1)
                else:
                    return r**l / r_dash**(l - 1)

            for s in range(0, N - 2, 2):
                sum += (r[s + 2] - r[s]) * (fl(r[s], r[k]) * D2(s, l, m) +
                                            4 * fl(r[s + 1], r[k]) * D2(s + 1, l, m) +
                                            fl(r[s + 2], r[k]) * D2(s + 2, l, m))

            return sum / 6

        def calc_Phi(i, j, k):
            Phi = 0

            for l in range(lmax + 1):
                for m in range(min(l + 1, 2)):
                    if (m + l % 2 == 1):
                        continue
                    if m == 0:
                        eps = 1
                    else:
                        eps = 2
                    Phi -= eps / factorial(1 + m) * \
                        D3(l, m, k) * lpmv(m, l, mu[j]) * np.cos(m * ph[i])

            return Phi

        # calculate Phi across grid
        for n in range(N):
            for k in range(K):
                for m in range(M):
                    star.Phi[m, k, n] = calc_Phi(m, k, n)

        print(f'Phi = {star.Phi[0,0,:]}')

        # update the enthalpy

        Omega2 = star.eos.Omega2(star.Phi, star.Psi)
        C = star.eos.C(star.Phi, star.Psi)

        H = C - star.Phi - Omega2 * star.Psi

        # use new enthalpy and Phi to calculate the density

        star.rho = star.eos.rho_from_h(H)
        star.rho /= np.max(star.rho)

        print(f"rho = {np.average(star.rho[:,0,:], axis=0)}")

        # make sure density is always non-negative
        # star.rho[star.rho < 0] = 0

        print(f"rho = {np.average(star.rho[:,0,:], axis=0)}")

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
                                                        4 *
                                                        star.rho[i + 1, j, k]
                                                        + star.rho[i + 2, j, k])

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
