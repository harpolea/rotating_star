import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_raises
import pytest

from star import Star
import eos
import solvers
import rotation_laws


class TestStar(object):
    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        pass

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup_method(self):
        """ this is run before each test """
        self.rotation_law = "rigid"
        self.eos = "polytrope"
        self.solver = "SCF"

        self.mesh_size = (3, 3)
        self.parameters = {'K': 1, 'N': 0, 'A': (
            2, 0, 0), 'B': (1, 0, 0), 'Omega0': 1}

        self.star = Star(self.rotation_law, self.eos,
                         self.solver, self.mesh_size)
        self.star.initialize_star(self.parameters)

    def teardown_method(self):
        """ this is run after each test """
        pass

    def test_object_initialization(self):
        assert isinstance(self.star.eos, eos.Polytrope)
        assert isinstance(self.star.rotation_law, rotation_laws.RigidRotation)
        assert isinstance(self.star.solver, solvers.SCF)

        assert np.shape(self.star.rho) == self.star.mesh_size
        assert np.shape(self.star.Phi) == self.star.mesh_size

        assert self.star.eos.N == self.parameters['N']
        assert self.star.eos.K == self.parameters['K']


class TestEOS(object):
    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        cls.parameters = {'K': 1, 'N': 1e9,
                          'A': (2, 0, 0), 'B': (1, 0, 0),
                          'a': 1, 'b': 1, 'x': 1}

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup_method(self):
        """ this is run before each test """
        pass

    def teardown_method(self):
        """ this is run after each test """
        pass

    def test_polytrope(self):
        poly = eos.Polytrope()

        rho = np.array([1e-5, 1, 1e3])
        p = np.array([1e-5, 1, 1e3])
        h = np.array([1 + 1e9, 1 + 1e9, 1 + 1e9])

        assert_raises(Exception, poly.p_from_rho, rho)

        poly.initialize_eos(self.parameters)

        assert_allclose(p, poly.p_from_rho(rho))
        assert_allclose(h, poly.h_from_rho(rho))

        poly.N = 1.5
        poly.K = 1e-3

        p = np.array([4.641588834e-12, 1e-3, 1e2])
        h = np.array([1.160397208e-6, 2.5e-3, 0.25])

        assert_allclose(p, poly.p_from_rho(rho))
        assert_allclose(h, poly.h_from_rho(rho))

    def test_wd(self):
        wd = eos.WD_matter()

        rho = np.array([0, 1, 1e3])
        p = np.ones_like(rho) * 1.229907199
        h = np.array([8, 11.3137085, 80.39900497])

        assert_raises(Exception, wd.p_from_rho, rho)

        wd.initialize_eos(self.parameters)

        assert_allclose(p, wd.p_from_rho(rho))
        assert_allclose(h, wd.h_from_rho(rho))

        wd.a = 2
        wd.x = 5

        p = np.ones_like(rho) * 2410.413801
        h = np.array([16, 22.627417, 160.798009])

        assert_allclose(p, wd.p_from_rho(rho))
        assert_allclose(h, wd.h_from_rho(rho))
