import numpy as np

import eos
from scf import SCF


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

    def initialize_star(self, parameters):
            self.eos.initialize_eos(parameters)

    def solve_star(self):
        # make a guess for rho
        self.rho[:,:,:] = 1

        scf = SCF(self)
        scf.solve()



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
