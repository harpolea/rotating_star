from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import lpmv, factorial, eval_legendre
from scipy.integrate import simps, odeint
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


class Roxburgh(Solver):

    def initialize_solver(self, parameters):
        self.theta_m = 0
        self.Ro = 0.9

        self.Rs = self.star.r_coords[-1]

        # inverted coordinate system is driving me mad so I am .T'ing them all.
        self.star.rho = self.star.rho.T
        self.star.Phi = self.star.Phi.T
        self.star.Omega2 = parameters['Omega']**2

        self.rhom = self.star.rho[:, 0]

    def step(self):
        # solve Poisson for Phi given rho
        Phi = self.Phi_given_rho(self.star.rho)

        # solve for rho given Phi subject to rho(r,0) = rhom(r)
        rho = self.rho_given_Phi

        # test conversion
        rho_err = np.max(np.abs(rho - self.star.rho)) / np.max(rho)

        print(f"Error: rho_err = {rho_err}")

        # print(f"rho = {rho}")
        # print(f"Phi = {Phi}")

        self.star.rho = rho
        self.star.Phi = Phi

        return rho_err

    def Phi_given_rho(self, rho):
        r = self.star.r_coords
        th = self.star.theta_coords
        nth = len(th)
        nr = len(r)

        nk = min(20, nth)

        Phi = np.zeros_like(self.star.Phi)

        W = np.array([[eval_legendre(2 * k, np.cos(th[n]))
                       for n in range(nk)] for k in range(nk)])

        ck = np.zeros((nr, nk))
        fk = np.zeros((nr, nk))

        for i in range(nr):
            ck[i, :] = np.linalg.matmul(np.linalg.inv(W), rho[i, :nk])

        for k in range(nth):
            # solve equation 12 using shooting

            def dfdr(x, _r):
                y = np.zeros_like(x)
                y[:, 0] = 4 * np.pi * self.star.G * ck[:, k] - 2 / \
                    _r * x[:, 0] + 2 * k * (k + 1) * x[:, 0] / _r**2
                y[:, 1] = x[:, 0]

                return y

            def shoot_me(dfdr0):
                _fk = odeint(dfdr, [dfdr0, 0], r)

                # calculate boundary condition at r=Rs
                return (2 * k + 1) * _fk[-1, 1] + r[-1] * _fk[-1, 0]

            dfdr0 = brentq(shoot_me, -1, 1)

            fk[:, k] = odeint(dfdr, [dfdr0, 0], r)[:, 1]

        # now given fk we can find Phi
        for i in range(nr):
            for j in range(nth):
                Phi[i, j] = np.sum(
                    [fk[i, k] * eval_legendre(2 * k, np.cos(th[j])) for k in range(nk)])

        return Phi

        # for i in range(nr):
        #
        #     lk = np.zeros(nth)
        #     for k in range(nth):
        #
        #         I = ck[:, k] * r**(2*k+2)
        #
        #         lk[k] = 4 * np.pi * self.star.G / ((4 * k + 1) * self.Rs**(4*k+1)) * simps(I, r)
        #
        #         inner_I = np.array([simps(c[:j, k] * r[:j]**(2*k+2)) for k in range(nr)])
        #
        #         I2 = 4 * np.pi * self.star.G / r[i:]**(4*k+2)

    def rho_given_Phi(self, Phi):
        r = self.star.r_coords
        th = self.star.theta_coords
        dr = r[1] - r[0]
        dth = th[1] - th[0]
        nth = len(th)
        nr = len(r)

        rho = np.zeros_like(Phi)

        dPhidr = np.zeros_like(Phi)
        dPhidth = np.zeros_like(Phi)

        dPhidr[1:-1, :] = (Phi[2:, :] - Phi[:-2, :]) / (2 * dr)
        dPhidth[:, 1:-1] = (Phi[:, 2:] - Phi[:, :-2]) / (2 * dth)

        # do boundaries by copying. At r=0, we want the derivative of the potential to be 0, so we'll leave that.
        dPhidr[-1, :] = dPhidr[-2, :]

        dPhidth[:, 0] = dPhidth[:, 1]
        dPhidth[:, -1] = dPhidth[:, -2]

        gm = np.zeros_like(Phi)

        for i in range(nr):
            for j in range(nth):

                I = -1 / r[i] * (dPhidth[i, :j] - self.star.Omega2 * r[i]**2 * np.sin(th[:j]) * np.cos(
                    th[:j])) / (dPhidr[i, :j] - self.star.Omega2 * r[i] * np.sin(th[:j]**2))

                log_gm = simps(I, th[:j])

                gm[i, j] = np.exp(log_gm)

            for j in range(nth):

                dgdth = -1 / r[i] * (dPhidth[i, :j] - self.star.Omega2 * r[i]**2 * np.sin(
                    th[:j]) * np.cos(th[:j])) / (dPhidr[i, :j] - self.star.Omega2 * r[i] * np.sin(th[:j]**2))

                I = -r[i] * self.star.Omega2 * np.cos(th[:j]) / (
                    dPhidr[i, :j] - self.star.Omega2 * r[i] * np.sin(th[:j])**2) * 1 / dgdth

                log_rho = simps(I, gm[i, :j]) + np.log(self.rhom[i])

                rho[i, j] = np.exp(log_rho)

        return rho

    def P_given_rho_Phi(self, rho, Phi):
        r = self.star.r_coords
        th = self.star.theta_coords
        dr = r[1] - r[0]
        dth = th[1] - th[0]
        nth = len(th)
        nr = len(r)

        dPhidr = np.zeros_like(Phi)

        dPhidr[1:-1, :] = (Phi[2:, :] - Phi[:-2, :]) / (2 * dr)
        dPhidr[-1, :] = dPhidr[-2, :]

        P = np.zeros((nr, nth))

        # NOTE: used the fact that theta_m = 0

        Ro_index = self.star.eos.A[1]

        for i in range(Ro_index):
            P[i, 0] = np.simps(rho[i:Ro_index, 0] *
                               dPhidr[i:Ro_index, 0], r[i:Ro_index])

        # now we have to do something clever and interpolate these along the characteristics. eurgh.

    def solve(self, max_steps=100, delta=1e-3):

        for i in range(max_steps):
            print(f"Step {i}")
            rho_err = self.step()

            if rho_err <= delta:
                print("Solution found!")
                break

        self.star.p = self.P_given_rho_Phi(self.star.rho, self.star.Phi)

        # untranspose

        self.star.rho = self.star.rho.T
        self.star.Phi = self.star.Phi.T
        self.star.p = self.star.p.T

    def calc_mass(self):
        pass

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

        def K(self, K0, r, th):
            # it has a really weird error if this line is not in there
            if not hasattr(th, "__len__"):
                a = float(th)

            return K0 * (1 + self.eps * (np.sin(th)**2 / self.a0**2 + np.cos(th)**2 / self.b0**2) * r**self.m)

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

        self.star.eos.initialize_eos(parameters)
        self.star.rotation_law.initialize_law(parameters)

        self.q = self.star.r_coords[self.star.eos.A[1]]  # / self.star.eos.B[1]

        r2d = np.zeros(self.star.mesh_size)
        th2d = np.zeros_like(r2d)

        r2d[:, :] = self.star.r_coords[np.newaxis, :]
        th2d[:, :] = self.star.theta_coords[:, np.newaxis]

        self.star.K0 = 1e-3
        self.star.p = self.star.eos.p_from_rho(
            self.star.rho, self.star.K0, r2d, th2d)

    def step(self):

        r = self.star.r_coords
        th = self.star.theta_coords

        dr = r[1] - r[0]
        dth = th[1] - th[0]

        nth, nr = self.star.mesh_size

        # rho = self.star.rho
        rho = np.zeros((nth, nr))
        p = np.zeros((nth, nr))

        rho_old = self.star.rho
        p_old = self.star.p

        lmax = 20

        def fl(r_dash, r, l):
            if r_dash < r:
                return r_dash**(2 * l) / r**(2 * l + 1)
            else:
                return r**(2 * l) / r_dash**(2 * l + 1)

        def I1(l, i):
            integrand = eval_legendre(
                2 * l, np.cos(th)) * np.sin(th) * rho_old[:, i]

            return simps(integrand, th)

        def I2(l, i):
            if r[i] == 0:
                return 0
            else:
                integrand = r**2 * \
                    np.array([fl(r[i], r_dash, l) * I1(l, k)
                              for k, r_dash in enumerate(r)])

                return simps(integrand, r)

        Phi = np.zeros_like(rho)

        for j in range(nth):
            for i in range(nr):
                for l in range(lmax):
                    Phi[j, i] -= eval_legendre(2 * l, np.cos(th[j])) * I2(l, i)
        Phi[:, 0] = Phi[:, 1]

        # now find j0, K0

        # start by looking at the rotational axis, j=0
        rho_c = 1
        p_c = self.star.eos.p_from_rho(rho_c, self.star.K0, 0, 0)

        def shoot_me(K0):

            rho[:, :] = 0
            p[:, :] = 0

            p_c = self.star.eos.p_from_rho(rho_c, K0, 0, 0)

            def root_find_me(_rho, rho_m, p_m, i):
                _p = self.star.eos.p_from_rho(_rho, K0, r[i], 0.0)

                # print(f"_rho = {_rho}, _p = {_p}, rho_m = {rho_m}, p_m = {p_m}, deltaPhi = {Phi[0,i] - Phi[0,i-1]}")

                return (_p - p_m) / dr - (Phi[0, i] - Phi[0, i - 1]) * (_rho + rho_m) / (2 * dr)

            # find centre values
            # TODO: what on earth is delta phi supposed to be here????
            rho[:, :] = self.star.rho[:, :]
            rho[:, 0] = rho_c
            p[:, :] = self.star.p[:, :]
            p[:, 0] = self.star.eos.p_from_rho(rho[0, 0], K0, 0, 0)

            r_pol = 0

            for i in range(1, nr):

                # print(f"root_find_me(0) = {root_find_me(0, rho[0, i - 1], p[0, i - 1], i)}, root_find_me(rho[0, i - 1])  = {root_find_me(rho[0, i - 1], rho[0, i - 1], p[0, i - 1], i)}")

                if root_find_me(0, rho[0, i - 1], p[0, i - 1], i) * root_find_me(rho[0, i - 1] * 1.05, rho[0, i - 1], p[0, i - 1], i) > 0:
                    r_pol = r[i]
                    break

                rho[0, i] = brentq(root_find_me, 0, rho[0, i - 1] * 1.05, args=(
                    rho[0, i - 1], p[0, i - 1], i))

                if (rho[0, i] <= 0 or r[i] >= 1):
                    r_pol = r[i]
                    break

                p[0, i] = self.star.eos.p_from_rho(rho[0, i], K0, r[i], 0)

            return r_pol - self.q

        K_min = 1e-5
        K_max = 1

        if shoot_me(K_min) * shoot_me(K_max) > 0:
            K_max *= 1e6
            K_min *= 1e-2

        print(f"self.q = {self.q}")

        print(
            f"shoot_me({K_min}) = {shoot_me(K_min)}, shoot_me(0) = {shoot_me(0)}, shoot_me({K_max}) = {shoot_me(K_max)}")

        K0 = brentq(shoot_me, K_min, K_max)

        print(f"Found K0 = {K0}!")

        omega2 = self.star.rotation_law.omega2

        p_c = self.star.eos.p_from_rho(rho_c, K0, 0, 0)

        def shoot_j0(j0):

            rho[:, :] = 0
            p[:, :] = 0

            def root_find_me(_rho, rho_m, p_m, i):
                _p = self.star.eos.p_from_rho(_rho, K0, r[i], np.pi / 2)

                return (_p - p_m) / dr - \
                    0.5 / dr * (_rho + rho_m) * (Phi[-1, i] - Phi[-1, i - 1]) - \
                    0.5**3 * (r[i] + r[i - 1]) * (_rho + rho_m) * \
                    (omega2(j0, r[i]) + omega2(j0, r[i - 1]))

            # find centre values

            rho[:, 0] = rho_c
            p[:, 0] = p_c

            r_pol = 0

            for i in range(1, nr):

                if root_find_me(0, rho[-1, i - 1], p[-1, i - 1], i) * root_find_me(rho[-1, i - 1] * 1.05, rho[-1, i - 1], p[-1, i - 1], i) > 0:
                    r_pol = r[i - 1]
                    break

                rho[-1, i] = brentq(root_find_me, 0, rho[-1, i - 1] * 1.05,
                                    args=(rho[-1, i - 1], p[-1, i - 1], i))

                if (rho[-1, i] <= 0 or r[i] >= 1):
                    r_pol = r[i]
                    break

                p[-1, i] = self.star.eos.p_from_rho(
                    rho[-1, i], K0, r[i], np.pi / 2)

            return r_pol - self.q

        j0_min = 0
        j0_max = 1

        print(
            f"shoot_j0(0) = {shoot_j0(0)}, shoot_j0(1e-2) = {shoot_j0(1e-2)}, shoot_j0(1) = {shoot_j0(1)}")

        j0 = brentq(shoot_j0, 0, j0_max)

        Omega2 = np.zeros_like(rho)

        # first set the values of Omega2 on the equatorial plane

        Omega2[-1, :] = omega2(j0, r)

        # now we want to find the values of Omega2 elsewhere.
        # we do this by iterating backwards from the equatorial plane.
        for i in range(1, nr):
            rm = 0.5 * (r[i] + r[i - 1])
            for j in range(nth - 2, 0, -1):

                thm = 0.5 * (th[j] + th[j - 1])
                rhom = 0.5 * (rho_old[j, i - 1] + rho_old[j, i] +
                              rho_old[j + 1, i - 1] + rho_old[j + 1, i])

                if rhom == 0:
                    # outside the star
                    continue

                rhs = 0.25 / (dr * dth * rhom**2) * \
                    ((rho_old[j + 1, i] + rho_old[j + 1, i - 1] - rho_old[i, j] - rho_old[j, i - 1]) *
                     (p_old[j + 1, i] + p_old[j, i] - p_old[j + 1, i - 1] - p_old[j, i - 1]) -
                     (p_old[j + 1, i] + p_old[j + 1, i - 1] - p_old[j, i] - p_old[j, i - 1]) *
                     (rho_old[j + 1, i] + rho_old[j, i] - rho_old[j + 1, i - 1] - rho_old[j, i - 1]))
                lhs = 0.5 * rm**2 * np.sin(thm) * np.cos(thm) / dr * \
                    (Omega2[j + 1, i] - Omega2[j, i - 1] - Omega2[j + 1, i - 1]) - \
                    0.5 * rm * np.sin(thm)**2 / dth * \
                    (Omega2[j + 1, i] + Omega2[j + 1, i - 1] - Omega2[j, i - 1])

                coeff = 0.5 * rm**2 * np.sin(thm) * np.cos(thm) / dr + \
                    0.5 * rm * np.sin(thm)**2 / dth

                Omega2[j, i] = (rhs - lhs) / coeff

        # now finally find new p, rho
        p[:, 0] = p_c
        rho[:, 0] = rho_c

        print(f"j0 = {j0}, K0 = {K0}")

        def this_is_not_explicit(rho_guess, i, j):
            p_guess = self.star.eos.p_from_rho(rho_guess, K0, r[i], th[j])

            # print(
            # f"rho_guess = {rho_guess}, p_guess = {p_guess}, p_m = {p[j, i - 1]}")

            return (p_guess - p[j, i - 1]) / dr - \
                (0.5 / dr * (rho_guess + rho[j, i - 1]) * (Phi[j, i] - Phi[j, i - 1]) +
                 0.5**3 * (r[i] + r[i - 1]) * np.sin(th[j])**2 * (rho_guess + rho[j, i - 1]) *
                 (Omega2[j, i] + Omega2[j, i - 1]))

        for j in range(nth):
            for i in range(1, nr):
                # print(
                #     f"this_is_not_explicit(0) = {this_is_not_explicit(0,i,j)}, this_is_not_explicit(r_i-1) = {this_is_not_explicit(rho[j,i-1]*1.05,i,j)}")

                if this_is_not_explicit(0, i, j) * this_is_not_explicit(rho[j, i - 1], i, j) > 0:
                    # reached stellar surface so skip the rest
                    rho[j, i:] = 0
                    p[j, i:] = 0
                    break

                rho[j, i] = brentq(this_is_not_explicit, 0,
                                   rho[j, i - 1] * 1.05, args=(i, j))
                p[j, i] = self.star.eos.p_from_rho(rho[j, i], K0, r[i], th[j])

                if rho[j, i] <= 0:
                    # reached stellar surface so skip the rest
                    rho[j, i:] = 0
                    p[j, i:] = 0
                    break

        # check errors

        rho_err = np.max(np.abs(rho - rho_old)) / np.max(rho)

        print(f"Errors: rho_err = {rho_err}")

        # print(f"rho = {rho}")
        # print(f"Phi = {Phi}")

        self.star.rho = rho
        self.star.p = p
        self.star.K0 = K0

        return rho_err

    def solve(self, max_steps=100, delta=1e-3):

        for i in range(max_steps):
            print(f"Step {i}")
            rho_err = self.step()

            if rho_err <= delta:
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
