import numpy as np

import eos


class Star(object):

    def __init__(self, rotation_law, _eos):
        """
        Constructor
        """
        laws = {"rigid": self.rigid_rotation, "v-constant": self.v_constant_rotation,
                "j-constant": self.j_constant_rotation}
        self.rotation_law = laws[rotation_law]

        eoses = {"polytrope": eos.Polytrope, "wd": eos.WD_matter}

        self.eos = eoses[_eos]

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

    def SCF(self):
        """ Implements the self-consistent field algorithm """
