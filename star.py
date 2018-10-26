import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import zoom

import eos
from solvers import SCF, Newton, FSCF, Roxburgh
from rotation_laws import RigidRotation, VConstantRotation, JConstantRotation

plt.rcParams.update({'font.size': 20, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif'})


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
            self.eos = eoses[_eos]()
        except KeyError:
            raise KeyError(f"EoS must be one of: {eoses.keys()}")

        solvers = {"SCF": SCF, "Newton": Newton, "FSCF": FSCF, "Roxburgh":Roxburgh}
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

            self.Psi[:, :, :] = -0.5 * self.r_coords[np.newaxis, np.newaxis, :]**2 * \
                (1 - self.mu_coords[np.newaxis, :, np.newaxis]**2)
        else:

            self.theta_coords = np.linspace(0, np.pi/2, self.mesh_size[0], endpoint=True)

            self.r_coords = np.linspace(0, 2, self.mesh_size[1], endpoint=True)

            self.omegabar = np.zeros(self.mesh_size)
            self.omegabar[:, :] = self.r_coords[np.newaxis, :] * \
                np.sin(self.theta_coords[:, np.newaxis])

        self.rmax = self.r_coords[-1]

        self.H = np.zeros(self.mesh_size)
        self.Omega2 = 0
        self.C = 0
        self.M = 0
        self.W = 0

    def initialize_star(self, parameters):
        self.eos.initialize_eos(parameters)
        self.rotation_law.initialize_law(parameters)

        # make a guess for rho and Phi
        if self.dim == 3:
            self.rho[:, :, :] = 1 - self.r_coords[np.newaxis, np.newaxis, :]
            # print(f"rho = {self.rho[0,0,:]}")
        else:
            self.rho[:, :] = 1 - self.r_coords[np.newaxis, :]
            # print(f"rho = {self.rho[0,:]}")

        self.rho[self.rho < 0] = 0

        self.rho /= np.max(self.rho)

        self.H = self.eos.h_from_rho(self.rho)

        self.solver.initialize_solver(parameters)

    def solve_star(self, max_steps=100, delta=1e-3):

        self.solver.solve(max_steps, delta)

        # find mass and gravitational energy
        M = self.solver.calc_mass()
        W = self.solver.calc_gravitational_energy()

        self.M = M
        self.W = W
        #
        # return M, W

    def plot_star(self):
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 10))

        rB = self.r_coords[self.eos.B[-1]]
        rA = self.r_coords[self.eos.A[-1]]

        for ax in axes:
            ax.axvline(x=rA, linestyle=':', color='lightgrey')
            ax.axvline(x=rB, linestyle=':', color='lightgrey')

        # axes[0].text(rA, 1.1 * np.max(self.rho), r"$r_A$")
        # axes[0].text(rB, 1.1 * np.max(self.rho), r"$r_B$")

        if rA == rB:
            axes[0].text(rA, 1.1 * self.rho[0, 0], r"$r_{A,B}$")
        else:
            axes[0].text(rA, 1.1 * self.rho[0, 0], r"$r_A$")
            axes[0].text(rB, 1.1 * self.rho[0, 0], r"$r_B$")

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
        # axes[0].set_ylim([-0.05, 1.05])

        r_lim = max(self.eos.A[1], self.eos.B[1])

        axes[1].set_ylabel(r'$\Phi$')
        axes[1].set_ylim([np.min(self.Phi[:, :r_lim]),
                          1.05 * np.max(self.Phi[:, :r_lim])])
        axes[2].set_ylabel(r'$H$')
        axes[2].set_ylim([0, 1.05 * np.max(self.H[:, :r_lim])])

        axes[2].set_xlabel(r'$r$')

        fig.subplots_adjust(hspace=0)
        axes[2].set_xlim([0, 1.05])

        axes[2].legend()

        plt.show()

    def plot_isosurfaces(self, nlevels=10):
        """ Plot surfaces of equal density/pressure/potential """

        fig, ax = plt.subplots(figsize=(8, 8))

        p = self.eos.p_from_rho(self.rho)

        fields = [self.rho, p, self.Phi]
        labels = [r"$\rho$", r"$p$", r"$\Phi$", 'surface']
        linestyles = ['-', ':', '--']

        Y = self.r_coords[np.newaxis, :] * \
            np.sin(self.theta_coords[:, np.newaxis])
        X = self.r_coords[np.newaxis, :] * \
            np.cos(self.theta_coords[:, np.newaxis])

        # first find out where the surface of the star is
        mask = ((self.rho >= 0) & (X**2 + Y**2 <= 1))

        cmap = plt.cm.tab10

        custom_lines = []
        for i, f in enumerate(fields):
            colour = cmap(i)
            iso_values = np.linspace(np.min(f[mask]), np.max(
                f[mask]), num=nlevels, endpoint=True)

            f_masked = np.zeros_like(f)
            f_masked[:, :] = f[:, :]
            f_masked[~mask] = np.ma.masked
            ax.contour(X, Y, f, colors=[
                       colour], levels=iso_values, linestyles=linestyles[i], linewidths=2, extend='both')

            custom_lines.append(
                Line2D([0], [0], color=colour, linestyle=linestyles[i], linewidth=2))

        # smooth rho to find smooth surface
        factor = 10
        ax.contour(zoom(X, factor), zoom(Y, factor), zoom(
            self.rho, factor), colors='k', levels=1e-3, linewidths=3)
        custom_lines.append(Line2D([0], [0], color='k', linewidth=3))

        ax.legend(custom_lines, labels)

        ax.set_xlim([0, self.r_coords[max(self.eos.A[1], self.eos.B[1])]])
        ax.set_ylim([0, self.r_coords[max(self.eos.A[1], self.eos.B[1])]])

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        plt.show()
