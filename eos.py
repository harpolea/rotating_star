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


class Polytrope(EOS):
    """ Polytrope equation of state """

    # def __init__(self):
    #     self.initialized = False

    def initialize_eos(self, parameters):
        self.initialized = True
        self.K = parameters['K']
        self.N = parameters['N']

    def p_from_rho(self, rho):
        """ eq4 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return self.K * rho ** (1 + 1 / self.N)

    def rho_from_h(self, h):
        """ eq 10 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return (h / (self.K * (1 + self.N)))**self.N

    def h_from_rho(self, rho):
        """ eq 5 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return (1 + self.N) * self.p_from_rho(rho) / rho


class WD_matter(EOS):
    """ Equation of state for a white dwarf with zero temperature """

    # def __init__(self):
    #     self.initialized = False

    def initialize_eos(self, parameters):
        self.initialized = True
        self.a = parameters['a']
        self.b = parameters['b']
        self.x = parameters['x']

    def p_from_rho(self, rho):
        """ eq 6 """
        if not self.initialized:
            raise Exception("EOS not initialized")

        return self.a * (self.x * (2 * self.x**2 - 3) * np.sqrt(self.x**2 + 1) +
                         3 * np.arcsinh(self.x))

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
