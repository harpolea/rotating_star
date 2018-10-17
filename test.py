import numpy as np
from numpy.testing import assert_array_equal
import pytest

from star import Star
import eos


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

        self.mesh_size = (3, 3, 3)
        self.parameters = {'K': 1, 'N': 0, 'A': (2, 0, 0), 'B': (1, 0, 0)}

        self.star = Star(self.rotation_law, self.eos,
                         self.solver, self.mesh_size)
        self.star.initialize_star(self.parameters)

    def teardown_method(self):
        """ this is run after each test """
        pass

    def test_object_initialization(self):
        assert self.star.eos == eos.Polytrope
