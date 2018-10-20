from abc import ABCMeta, abstractmethod
import numpy as np


class EOS(metaclass=ABCMeta):
    # equation of state class

    def __init__(self):
        """ constructor """
        self.initialized = False

    @abstractmethod
    def initialize_eos(self, parameters):
        pass

    @abstractmethod
    def p_from_rho(self, rho):
        pass

    @abstractmethod
    def rho_from_h(self, h):
        pass

    @abstractmethod
    def h_from_rho(self, rho):
        pass

    @abstractmethod
    def Omega2(self, Phi, Psi):
        pass

    @abstractmethod
    def C(self, Phi, Psi):
        pass

    @abstractmethod
    def rho_H_dash(self, h):
        pass


class Polytrope(EOS):
    """ Polytrope equation of state """

    def initialize_eos(self, parameters):
        self.initialized = True
        self.K = parameters['K']
        self.N = parameters['N']
        self.A = parameters['A']
        self.B = parameters['B']

    def p_from_rho(self, rho):
        """ eq4 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        if self.N == 0:
            return rho
        else:
            return self.K * rho ** (1 + 1 / self.N)

    def rho_from_h(self, h):
        """ eq 10 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return (h / (self.K * (1 + self.N)))**self.N

        rho = np.zeros_like(h)

        rho[h >= 0] = (h[h >= 0] / (self.K * (1 + self.N)))**self.N

        return rho

    def h_from_rho(self, rho):
        """ eq 5 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        if isinstance(rho, np.ndarray):
            h = np.zeros_like(rho)

            h[rho > 0] = (1 + self.N) * \
                self.p_from_rho(rho[rho > 0]) / rho[rho > 0]
        else:
            h = (1 + self.N) * self.p_from_rho(rho) / rho

        return h

    def Omega2(self, Phi, Psi):
        """ eq 16 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        if abs(Psi[self.A] - Psi[self.B]) < 1e-9:
            return 0
        else:
            return - (Phi[self.A] - Phi[self.B]) / (Psi[self.A] - Psi[self.B])

    def C(self, Phi, Psi):
        """ eq 17 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return Phi[self.A] + self.Omega2(Phi, Psi) * Psi[self.A]

    def rho_H_dash(self, h):
        if not self.initialized:
            raise Exception("EOS not initialized")

        return self.N * h**(self.N - 1) / (self.K * (1 + self.N))**self.N

        rho = np.zeros_like(h)
        rho[h < 0] = 0

        rho[h >= 0] = self.N * \
            h[h >= 0]**(self.N - 1) / (self.K * (1 + self.N))**self.N

        return rho


class WD_matter(EOS):
    """ Equation of state for a white dwarf with zero temperature """

    def initialize_eos(self, parameters):
        self.initialized = True
        self.a = parameters['a']
        self.b = parameters['b']
        self.x = parameters['x']
        self.A = parameters['A']
        self.B = parameters['B']

    def p_from_rho(self, rho):
        """ eq 6 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return self.a * (self.x * (2 * self.x**2 - 3) * np.sqrt(self.x**2 + 1)
                         + 3 * np.arcsinh(self.x))

    def rho_from_h(self, h):
        """ eq 11 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return self.b * ((self.b * h / (8 * self.a))**2 - 1)**1.5

    def h_from_rho(self, rho):
        """ eq 7 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return (8 * self.a / self.b) * (1 + (rho / self.b)**(2 / 3))**0.5

    def Omega2(self, Phi, Psi):
        """ eq 20 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return - (Phi[self.A] - Phi[self.B]) / (Psi[self.A] - Psi[self.B])

    def C(self, Phi, Psi):
        """ eq 21 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return 8 * self.a / self.b + Phi[self.A] + \
            self.Omega2(Phi, Psi) * Psi[self.A]

    def rho_H_dash(self, h):
        raise NotImplementedError("This has not been implemented")
