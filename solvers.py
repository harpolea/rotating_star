from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import lpmv, factorial, eval_legendre
from scipy.integrate import simps
from scipy.optimize import brentq

from eos import Polytrope
from rotation_laws import JConstantRotation


class Solver(metaclass=ABCMeta):

    def __init__(self, star):
        self.star = star
        self.initialized = False

    @abstractmethod
    def initialize_solver(self, parameters):
        pass

    @abstractmethod
    def solve(self, max_steps=100, delta=1e-3):
        pass

    @abstractmethod
    def calc_mass(self):
        pass

    @abstractmethod
    def calc_gravitational_energy(self):
        pass


class SCF(Solver):

    def initialize_solver(self, parameters):
        self.star.r_coords = np.array(
            range(1, self.star.mesh_size[1] + 1)) / (self.star.mesh_size[1] - 1)

        self.star.mu_coords = np.array(
            range(self.star.mesh_size[0])) / (self.star.mesh_size[0] - 1)

        self.star.Psi = np.zeros(self.star.mesh_size)
        self.star.Psi[:, :] = -0.5 * self.star.r_coords[np.newaxis, :]**2 * \
            (1 - self.star.mu_coords[:, np.newaxis]**2)
        self.star.omegabar = np.sqrt(-2 * self.star.Psi)

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

    def solve(self, max_steps=100, delta=1e-4):
        """
        Iterate single step until errors in enthalpy, spin rate and C are small
        """
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
                          (star.rho[:-2:2, j] + 4 * star.rho[1:-1:2, j] +
                           star.rho[2::2, j])) / 6

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
                                                    star.rho[2::2, j] * star.Phi[2::2, j])) / 6

        W = 0

        for j in range(0, N - 2, 2):
            W += (r[j + 2] - r[j]) * (r[j]**2 * S1(j) +
                                      4 * r[j + 1]**2 * S1(j + 1) +
                                      r[j + 2]**2 * S1(j + 2))

        return -1 / 3 * np.pi * W


class Newton(Solver):
    """ Implements the Newton solver of Aksenov and Blinnikov """

    def initialize_solver(self, parameters):
        self.initialized = True

        self.rho0 = parameters['rho0']

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
        self.star.rho *= self.rho0 / self.star.rho[-1, 0]

        print(f"rho = {self.star.rho[0,:]}")
        print(f"H = {H[0,:]}")

    def H_from_Phi(self, Phi):
        if not self.initialized:
            raise Exception("solver not initialized")
        A = self.star.eos.A
        B = self.star.eos.B
        Chi = self.Chi

        C_Psi = (Phi[B] - Phi[A]) / (Chi[A] - Chi[B])

        Psi = C_Psi * Chi

        C = 0.5 * (Phi[A] + Phi[B] + + Psi[A] + Psi[B])

        # print(f"C = {C}, Phi[A] = {Phi[A]}. Phi[B] = {Phi[B]}, Psi[A] = {Psi[A]}, Psi[B] = {Psi[B]}")

        return C - Phi - Psi

    def step(self):
        if not self.initialized:
            raise Exception("solver not initialized")

        A = self.star.eos.A
        B = self.star.eos.B

        mesh_size = self.star.mesh_size

        Phi = self.star.Phi
        Phi0 = Phi[0, 0]

        Chi = self.Chi
        Chi0 = Chi[0, 0]
        H = self.star.H  # _from_Phi(Phi)
        # H[A] = 0
        # H[B] = 0
        H0 = H[0, 0]

        rho = self.star.rho  # eos.rho_from_h(H)
        # rho[rho < 0] = 0
        # rho /= np.max(rho[0,0])

        rho0 = self.rho0  # rho[0, 0]  # rho[0, 0]
        # H0 = self.star.eos.h_from_rho(rho0)

        C_Psi = (Phi[B] - Phi[A]) / (Chi[A] - Chi[B])

        Psi = C_Psi * Chi
        Psi0 = Psi[0, 0]

        C = 0.5 * (Phi[A] + Phi[B] + Psi[A] + Psi[B])

        # C = Phi[A] + Psi[A]  # - (Phi0 + Psi0) * H[A] / H0) / (1 - H[A] / H0)

        _B = C - Phi - Psi
        B0 = _B[0, 0]
        # rho = self.star.eos.rho_from_h(H) / self.rho0 * \
        #     (H0 / (C - Phi0 - Psi0) * (C - Phi - Psi))

        # print(f"B0 = {B0}, Phi0 = {Phi0}, Psi0 = {Psi[0,0]}, C = {C}")
        # print(f"B = {_B}")

        rho_H_dash = self.star.eos.rho_H_dash(H)

        gPhi = 4 * np.pi * self.star.G * rho_H_dash * H0 / B0

        gA = - gPhi * ((Chi[A] - Chi) / (Chi[A] - Chi[B]) -
                       _B / B0 * (Chi[A] - Chi0) / (Chi[A] - Chi[B]))

        gB = - gPhi * ((Chi - Chi[B]) / (Chi[A] - Chi[B]) -
                       _B / B0 * (Chi0 - Chi[B]) / (Chi[A] - Chi[B]))

        g0 = - gPhi * _B / B0

        R = (4 * np.pi * self.star.G * rho + gA *
             Phi[A] + gB * Phi[B] + g0 * Phi0 + gPhi * Phi).flatten()

        r = self.star.r_coords
        th = self.star.theta_coords

        dr = r[1] - r[0]
        dth = th[1] - th[0]

        # calculate LHS matrix operator
        nx = mesh_size[0] * mesh_size[1]
        M = np.zeros((nx, nx))

        # del^2_r
        for j in range(mesh_size[0]):
            for i in range(1, mesh_size[1] - 1):
                ix = j * mesh_size[1] + i
                M[ix, ix - 1] += 1 / (dr**2 * r[i]**2) * \
                    0.5 * (r[i]**2 + r[i - 1]**2)
                M[ix, ix] -= 1 / (dr**2 * r[i]**2) * 0.5 * \
                    (2 * r[i]**2 + r[i - 1]**2 + r[i + 1]**2)
                M[ix, ix + 1] += 1 / (dr**2 * r[i]**2) * \
                    0.5 * (r[i]**2 + r[i + 1]**2)

        # del^2_theta
        for j in range(1, mesh_size[0] - 1):
            for i in range(1, mesh_size[1]):
                ix = j * mesh_size[1] + i
                jm = (j - 1) * mesh_size[1] + i
                jp = (j + 1) * mesh_size[1] + i

                M[ix, jm] += 1 / (r[i]**2 * np.sin(th[j]) * dth**2) * \
                    0.5 * (np.sin(th[j]) + np.sin(th[j - 1]))
                M[ix, ix] -= 1 / (r[i]**2 * np.sin(th[j]) * dth**2) * 0.5 * \
                    (2 * np.sin(th[j]) + np.sin(th[j - 1]) + np.sin(th[j + 1]))
                M[ix, jp] += 1 / (r[i]**2 * np.sin(th[j]) * dth**2) * \
                    0.5 * (np.sin(th[j]) + np.sin(th[j + 1]))

        # do boundaries

        # r = 0
        for j in range(mesh_size[0]):
            ix = j * mesh_size[1]

            for k in range(mesh_size[0]):
                kx = k * mesh_size[1] + 1

                c = 6 / dr**2 * (np.cos(max(0, dth * (k - 0.5))) -
                                 np.cos(min(0.5 * np.pi, dth * (k + 0.5))))
                M[ix, kx] += c
                M[ix, ix] -= c

        # upper r boundary
        for j in range(mesh_size[0]):
            # NOTE: DO NOT include this or the computer will restart itself
            # use first order accurate backwards second derivative
            i = mesh_size[1] - 1
            ix = j * mesh_size[1] + mesh_size[1] - 1
            M[ix, ix - 2] += 1 / (r[i]**2 * dr**2) * \
                0.5 * (r[i - 2]**2 + r[i - 1]**2)
            M[ix, ix - 1] -= 1 / (r[i]**2 * dr**2) * \
                0.5 * (r[i]**2 + 2 * r[i - 1]**2 + r[i - 2]**2)
            M[ix, ix] += 1 / (r[i]**2 * dr**2) * \
                0.5 * (r[i]**2 + r[i - 1]**2)

        for i in range(1, mesh_size[1]):
            # theta = 0
            ix = i
            jp = mesh_size[1] + i
            M[ix, ix] -= 4 / (r[i]**2 * dth**2)
            M[ix, jp] += 4 / (r[i]**2 * dth**2)

            # theta = pi/2
            ix = (mesh_size[0] - 1) * mesh_size[1] + i
            jm = (mesh_size[0] - 2) * mesh_size[1] + i
            M[ix, jm] += 2 / (r[i]**2 * dth**2)
            M[ix, ix] -= 2 / (r[i]**2 * dth**2)

        # add on other stuff

        # add on g's
        M[:, :] += np.diag(gPhi.flatten())

        A_idx = A[0] * mesh_size[1] + A[1]
        B_idx = B[0] * mesh_size[1] + B[1]

        M[:, A_idx] += gA.flatten()
        M[:, B_idx] += gB.flatten()
        M[:, 0] += g0.flatten()

        Phi = np.linalg.solve(M, R).reshape(mesh_size)

        H = self.H_from_Phi(Phi)

        rho = self.star.eos.rho_from_h(H)
        rho[rho < 0] = 0
        rho /= rho[0, 0]  # np.max(rho[:,min(A[1], B[1])])

        H = self.star.eos.h_from_rho(rho)

        # Phi /= rho[0,0]**(1/self.star.eos.N)

        # rho = self.star.eos.rho_from_h(H) \
        #     / self.rho0 * (H0 / (C - Phi0 - Psi0) * (C - Phi - Psi))

        C_Psi = (Phi[B] - Phi[A]) / (Chi[A] - Chi[B])

        Psi = C_Psi * Chi

        C = 0.5 * (Phi[A] + Phi[B] + Psi[A] + Psi[B])
        # C = (Phi[A] + Psi[A])

        # print(f"rho = {rho[0,:]}")

        idx = max(A[1], B[1])

        H_err = np.max(
            np.abs(H[:, :idx] - self.star.H[:, :idx])) / np.max(np.abs(H[:, :idx]))

        Phi_err = np.max(
            np.abs(Phi[:, :idx] - self.star.Phi[:, :idx])) / np.max(np.abs(Phi[:, :idx]))

        rho_err = np.max(
            np.abs(rho[:, :idx] - self.star.rho[:, :idx])) / np.max(np.abs(rho[:, :idx]))

        C_err = np.abs(C - self.star.C) / np.abs(C)

        # set variables to new values

        self.star.H = H
        self.star.Phi = Phi
        self.star.rho = rho
        self.star.C = C

        print(
            f"\tErrors: H_err = {H_err}, C_err = {C_err}, rho_err = {rho_err}")

        return H_err, C_err, rho_err

    def solve(self, max_steps=100, delta=1e-3):
        if not self.initialized:
            raise Exception("solver not initialized")

        for i in range(max_steps):
            print(f"Step {i}")
            H_err, C_err, rho_err = self.step()

            if C_err < delta or (H_err < delta and rho_err < delta):
                print("Solution found!")
                break

        if i == max_steps - 1:
            print("No solution found :( Solve terminated as max_steps reached")

    def calc_mass(self):
        if not self.initialized:
            raise Exception("solver not initialized")

        mesh_size = self.star.mesh_size
        r = self.star.r_coords
        th = self.star.theta_coords

        deltaV = np.zeros(mesh_size)

        for j in range(mesh_size[0]):
            for i in range(mesh_size[1]):
                deltaV[j, i] = np.pi / 6 * ((r[i] + r[min(i + 1, mesh_size[1] - 1)])**3 - (r[max(i - 1, 0)] + r[i])**3) * (
                    np.cos(0.5 * (th[max(j - 1, 0)] + th[j])) - np.cos(0.5 * (th[j] + th[min(j + 1, mesh_size[0] - 1)])))

        return np.sum(self.star.rho * deltaV)

    def calc_gravitational_energy(self):
        if not self.initialized:
            raise Exception("solver not initialized")

        mesh_size = self.star.mesh_size
        r = self.star.r_coords
        th = self.star.theta_coords

        deltaV = np.zeros(mesh_size)

        for j in range(mesh_size[0]):
            for i in range(mesh_size[1]):
                deltaV[j, i] = np.pi / 6 * ((r[i] + r[min(i + 1, mesh_size[1] - 1)])**3 - (r[max(i - 1, 0)] + r[i])**3) * (
                    np.cos(0.5 * (th[max(j - 1, 0)] + th[j])) - np.cos(0.5 * (th[j] + th[min(j + 1, mesh_size[0] - 1)])))

        return np.sum(0.5 * deltaV * self.star.rho * self.star.Phi)


class FSCF(Solver):
    """ SCF-based method described in Fujisawa 2015 """

    class VariablePolytrope(Polytrope):
        """ This is a polytrope but K is now a function of r, theta """

        def initialize_eos(self, parameters):
            self.initialized = True
            self.eps = parameters['eps']
            self.a0 = parameters['a0']
            self.b0 = parameters['b0']
            self.m = parameters['m']
            self.N = parameters['N']
            self.A = parameters['A']
            self.B = parameters['B']

        def K(self, K0, r, theta):
            return self.K0 * (1 + self.eps * (np.sin(theta)**2 / self.a0**2 + np.cos(theta)**2 / self.b0**2) * r**self.m)

        def p_from_rho(self, rho, K0, r, theta):
            if not self.initialized:
                raise Exception("EOS not initialized")

            if self.N == 0:
                return rho
            else:
                return self.K(K0, r, theta) * rho ** (1 + 1 / self.N)

        def rho_from_h(self, h, K0, r, theta):
            if not self.initialized:
                raise Exception("EOS not initialized")

            # return (h / (self.K * (1 + self.N)))**self.N

            rho = np.zeros_like(h)

            rho[h >= 0] = (
                h[h >= 0] / (self.K(K0, r, theta) * (1 + self.N)))**self.N

            return rho

        def rho_H_dash(self, h, K0, r, theta):
            if not self.initialized:
                raise Exception("EOS not initialized")

            # return self.N * h**(self.N - 1) / (self.K * (1 + self.N))**self.N

            rho = np.zeros_like(h)
            rho[h < 0] = 0

            rho[h >= 0] = self.N * \
                h[h >= 0]**(self.N - 1) / \
                (self.K(K0, r, theta) * (1 + self.N))**self.N

            return rho

    class VariableConstantRotation(JConstantRotation):

        def omega2(self, j0, r):
            return j0**2 / (1 + r**2 / self.d**2)**2

    def initialize_solver(self, parameters):
        self.star.eos = self.VariablePolytrope()
        self.star.rotation_law = self.VariableConstantRotation()

        self.q = self.star.eos.A[1]  # / self.star.eos.B[1]

    def solve(self, max_steps=100, delta=1e-3):

        r = self.star.r_coords
        th = self.star.theta_coords

        dr = r[1] - r[0]
        dth = th[1] - th[0]

        rho = self.star.rho
        rho_new = np.zeros_like(rho)
        p_new = np.zeros_like(rho)

        nth, nr = self.star.mesh_size

        lmax = 20

        def fl(r_dash, r, l):
            if r_dash < r:
                return r_dash**(l + 2) / r**(l + 1)
            else:
                return r**l / r_dash**(l - 1)

        def I1(l, i):

            integrand = eval_legendre(
                2 * l, np.cos(th)) * np.sin(th) * rho[:, i]

            return simps(integrand, th)

        def I2(l, i):
            integrand = r**2 * \
                np.array([fl(r[i], r_dash) * I1(l, k)
                          for k, r_dash in enumerate(r)])

            return simps(integrand, r)

        Phi = np.zeros_like(rho)

        for j in range(nth):
            for i in range(nr):
                for l in range(lmax):
                    Phi[j, i] -= eval_legendre(2 * l, np.cos(th[j])) * I2(l, i)

        # now find j0, K0

        # start by looking at the rotational axis, j=0
        rho_c = 1
        p_c = self.star.eos.p_from_rho(rho_c)

        def shoot_me(K0):

            rho_new[:, :] = 0
            p_new[:, :] = 0

            def root_find_me(rho, rho_m, p_m, r_i, delta_phi):
                p = self.star.eos.p_from_rho(rho, K0, r_i, 0)

                return 2 * (p - p_m) / delta_phi - (rho + rho_m)

            # find centre values
            # TODO: what on earth is delta phi supposed to be here????
            rho_new[0, 0] = brentq(root_find_me, 0, rho_c, args=(
                rho_c, p_c, 0, Phi[0, 2] - Phi[0, 1]))
            p_new[0, 0] = self.star.eos.p_from_rho(rho_new[0, 0], K0, 0, 0)

            r_pol = 0

            for i in range(1, nr):
                if (rho[0, i - 1] <= 0):
                    r_pol = r[i - 1]
                    break

                rho_new[0, i] = brentq(root_find_me, 0, rho_new[0, i - 1], args=(
                    rho_new[0, i - 1], p_new[0, i - 1], r[i], Phi[0, i] - Phi[0, i - 1]))

                p_new[0, 0] = self.star.eos.p_from_rho(
                    rho_new[0, i], K0, r[i], 0)

            return r_pol - self.q

        K0 = brentq(shoot_me, 0, 1)

        

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

    def solve(self, max_steps=100, delta=1e-3):
        """
        Iterate single step until errors in enthalpy, spin rate and C are small
        """

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
