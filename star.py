import numpy as np
from matplotlib import pyplot as plt

import eos
from solvers import SCF


class Star(object):

    def __init__(self, rotation_law, _eos, solver, mesh_size, rmax=1.05, G=1):
        """
        Constructor
        """
        laws = {"rigid": self.rigid_rotation, "v-constant": self.v_constant_rotation,
                "j-constant": self.j_constant_rotation}
        try:
            self.rotation_law = laws[rotation_law]
        except KeyError:
            raise KeyError(f"Rotation law must be one of: {laws.keys()}")

        eoses = {"polytrope": eos.Polytrope, "wd": eos.WD_matter}
        try:
            self.eos = eoses[_eos]
        except KeyError:
            raise KeyError(f"EoS must be one of: {eoses.keys()}")

        solvers = {"SCF": SCF}
        try:
            self.solver = solvers[solver](self)
        except KeyError:
            raise KeyError(f"Solver must be one of: {solvers.keys()}")

        self.G = G

        self.mesh_size = mesh_size

        self.dim = len(mesh_size)

        # initialize mesh
        self.rho = np.zeros(self.mesh_size)
        self.Phi = np.zeros(self.mesh_size)

        # initialize coords

        if self.dim == 3:
            self.phi_coords = np.pi * \
                np.array(range(self.mesh_size[0])) / (self.mesh_size[0] - 1)

            self.mu_coords = np.array(
                range(self.mesh_size[1])) / (self.mesh_size[1] - 1)

            self.r_coords = np.array(
                range(1, self.mesh_size[2] + 1)) / (self.mesh_size[2] - 1)
        else:
            self.mu_coords = np.array(
                range(self.mesh_size[0])) / (self.mesh_size[0] - 1)

            self.r_coords = np.array(
                range(1, self.mesh_size[1] + 1)) / (self.mesh_size[1] - 1)


        self.rmax = self.r_coords[-1]

        self.Psi = np.zeros(self.mesh_size)
        if self.dim == 3:
            self.Psi[:, :, :] = -0.5 * self.r_coords[np.newaxis, np.newaxis,
                                                     :]**2 * (1 - self.mu_coords[np.newaxis, :, np.newaxis]**2)
        else:
            self.Psi[:, :] = -0.5 * self.r_coords[np.newaxis,
                                                  :]**2 * (1 - self.mu_coords[:, np.newaxis]**2)

        self.H = np.zeros(self.mesh_size)
        self.Omega2 = 0
        self.C = 0

    def initialize_star(self, parameters):
        self.eos.initialize_eos(self.eos, parameters)

        # make a guess for rho
        if self.dim == 3:
            self.rho[:, :, :] = 1 - self.r_coords[np.newaxis, np.newaxis, :]
            print(f"rho = {self.rho[0,0,:]}")
        else:
            self.rho[:, :] = 1 - self.r_coords[np.newaxis, :]
            print(f"rho = {self.rho[0,:]}")
        self.rho[self.rho < 0] = 0

        self.rho /= np.max(self.rho)

    def solve_star(self, max_steps=100):

        self.solver.solve(max_steps)

        # find mass and gravitational energy
        M = self.solver.calc_mass()
        W = self.solver.calc_gravitational_energy()

        return M, W

    def plot_star(self):
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8,10))

        if self.dim == 3:
            axes[0].plot(self.r_coords, self.rho[0, 0, :])
            axes[1].plot(self.r_coords, self.Phi[0, 0, :])
            axes[2].plot(self.r_coords, self.H[0, 0, :])

        else:
            axes[0].plot(self.r_coords, self.rho[0, :], marker='x')
            axes[1].plot(self.r_coords, self.Phi[0, :], marker='x')
            axes[2].plot(self.r_coords, self.H[0, :], marker='x')

        axes[0].set_ylabel(r'$\rho$')
        axes[1].set_ylabel(r'$\Phi$')
        axes[2].set_ylabel(r'$H$')

        axes[2].set_xlabel(r'$r$')

        fig.subplots_adjust(hspace=0)
        axes[2].set_xlim([0, 1])

        plt.show()

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
