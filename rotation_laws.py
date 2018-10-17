import numpy as np
from abc import ABCMeta, abstractmethod


class RotationLaw(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def initialize_law(self, parameters):
        pass

    @abstractmethod
    def omega2(self, omegabar):
        pass

    @abstractmethod
    def Chi(self, omegabar):
        pass


class RigidRotation(RotationLaw):

    def initialize_law(self, parameters):
        self.Omega0 = parameters['Omega0']

    def omega2(self, omegabar):
        return self.Omega0**2

    def Chi(self, omegabar):
        return -omegabar**2 / 2


class VConstantRotation(RotationLaw):

    def initialize_law(self, parameters):
        self.d = parameters['d']
        self.v0 = parameters['v0']

    def omega2(self, omegabar):
        return self.v0**2 / (self.d**2 + omegabar**2)

    def Chi(self, omegabar):
        return - 0.5 * np.log(self.d**2 + omegabar**2)

class JConstantRotation(RotationLaw):

    def initialize_law(self, parameters):
        self.d = parameters['d']
        self.j0 = parameters['j0']

    def omega2(self, omegabar):
        return self.j0**2 / (self.d**2 + omegabar**2)**2

    def Chi(self, omegabar):
        return 0.5 / (self.d**2 + omegabar**2)
