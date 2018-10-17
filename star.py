import numpy as np
from matplotlib import pyplot as plt

import eos
from solvers import SCF, Newton
from rotation_laws import RigidRotation, VConstantRotation, JConstantRotation


class Star(object):

    def __init__(self, rotation_law, _eos, solver, mesh_size, rmax=1.05, G=1):
        """
        Constructor
        """
        laws = {"rigid": RigidRotation, "v-constant": VConstantRotation,
                "j-constant": JConstantRotation}
        try:
            self.rotation_law = laws[rotation_law]()
        except KeyError:
            raise KeyError(f"Rotation law must be one of: {laws.keys()}")

        eoses = {"polytrope": eos.Polytrope, "wd": eos.WD_matter}
        try:
            self.eos = eoses[_eos]
        except KeyError:
            raise KeyError(f"EoS must be one of: {eoses.keys()}")

        solvers = {"SCF": SCF, "Newton": Newton}
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

            if solver == "Newton":
                self.r_coords = np.array(
                    range(self.mesh_size[1])) / (self.mesh_size[1] - 2)
            else:
                self.r_coords = np.array(
                    range(1, self.mesh_size[1] + 1)) / (self.mesh_size[1] - 1)

        self.rmax = self.r_coords[-1]

        self.Psi = np.zeros(self.mesh_size)
        if self.dim == 3:
            self.Psi[:, :, :] = -0.5 * self.r_coords[np.newaxis, np.newaxis, :]**2 * \
                (1 - self.mu_coords[np.newaxis, :, np.newaxis]**2)
        else:
            self.Psi[:, :] = -0.5 * self.r_coords[np.newaxis, :]**2 * \
                (1 - self.mu_coords[:, np.newaxis]**2)

        self.omegabar = np.sqrt(-2*self.Psi)

        self.H = np.zeros(self.mesh_size)
        self.Omega2 = 0
        self.C = 0
        self.M = 0
        self.W = 0

    def initialize_star(self, parameters):
        self.eos.initialize_eos(self.eos, parameters)
        self.rotation_law.initialize_law(parameters)
        self.solver.initialize_solver(parameters)

        # make a guess for rho and Phi
        if self.dim == 3:
            self.rho[:, :, :] = 1 - self.r_coords[np.newaxis, np.newaxis, :]
            print(f"rho = {self.rho[0,0,:]}")
        else:
            self.rho[:, :] = 1 - self.r_coords[np.newaxis, :]
            print(f"rho = {self.rho[0,:]}")

            r2d = np.zeros(self.mesh_size)
            r2d[:,:] = self.r_coords[np.newaxis,:]

            self.Phi[r2d < 1] = -1.5
            self.Phi[r2d == 1] = -1
            self.Phi[r2d > 1] = -0.5

        self.rho[self.rho < 0] = 0

        self.rho /= np.max(self.rho)

        self.H = self.eos.h_from_rho(self.eos, self.rho)


    def solve_star(self, max_steps=100):

        self.solver.solve(max_steps)

        # find mass and gravitational energy
        M = self.solver.calc_mass()
        W = self.solver.calc_gravitational_energy()

        self.M = M
        self.W = W

        return M, W

    def plot_star(self):
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 10))

        rB = self.r_coords[self.eos.B[-1]]
        rA = self.r_coords[self.eos.A[-1]]

        for ax in axes:
            ax.axvline(x=rA, linestyle=':', color='lightgrey')
            ax.axvline(x=rB, linestyle=':', color='lightgrey')

        axes[0].text(rA, 1.1, r"$r_A$")
        axes[0].text(rB, 1.1, r"$r_B$")

        if self.dim == 3:
            axes[0].plot(self.r_coords, self.rho[0, 0, :])
            axes[1].plot(self.r_coords, self.Phi[0, 0, :])
            axes[2].plot(self.r_coords, self.H[0, 0, :])

        else:
            axes[0].plot(self.r_coords, self.rho[0, :],
                         marker='x', label=r'$\mu = 0$')
            axes[0].plot(self.r_coords, self.rho[-1, :],
                         marker='o', label=r'$\mu = 1$')
            axes[1].plot(self.r_coords, self.Phi[0, :],
                         marker='x', label=r'$\mu = 0$')
            axes[1].plot(self.r_coords, self.Phi[-1, :],
                         marker='o', label=r'$\mu = 1$')
            axes[2].plot(self.r_coords, self.H[0, :],
                         marker='x', label=r'$\mu = 0$')
            axes[2].plot(self.r_coords, self.H[-1, :],
                         marker='o', label=r'$\mu = 1$')

        axes[0].set_ylabel(r'$\rho$')
        axes[0].set_ylim([0, 1.05])

        axes[1].set_ylabel(r'$\Phi$')
        axes[2].set_ylabel(r'$H$')

        axes[2].set_xlabel(r'$r$')

        fig.subplots_adjust(hspace=0)
        axes[2].set_xlim([0, 1.05])

        axes[2].legend()

        plt.show()
