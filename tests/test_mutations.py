import unittest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from n_queens.genetic_algorithm import GeneticAlgorithm as ga


class TestMutations(unittest.TestCase):

    def test_swap(self):
        np.testing.assert_array_equal(ga.mutation_swap([1, 2, 3, 4, 5], 0, 1), np.array([2, 1, 3, 4, 5]))
        np.testing.assert_array_equal(ga.mutation_swap([1, 2, 3, 4, 5, 6], 1, 0), np.array([2, 1, 3, 4, 5, 6]))
        np.testing.assert_array_equal(ga.mutation_swap([1, 2, 3, 4, 5, 6, 7], 3, 6), np.array([1, 2, 3, 7, 5, 6, 4]))

    def test_insert(self):
        np.testing.assert_array_equal(ga.mutation_insert([1, 2, 3, 4, 5], 0, 2), np.array([1, 3, 2, 4, 5]))
        np.testing.assert_array_equal(ga.mutation_insert([1, 2, 3, 4, 5, 6], 0, 5), np.array([1, 6, 2, 3, 4, 5]))
        np.testing.assert_array_equal(ga.mutation_insert([1, 2, 3, 4, 5, 6, 7], 3, 5), np.array([1, 2, 3, 4, 6, 5, 7]))

    def test_scramble(self):
        np.testing.assert_equal(all(np.equal(ga.mutation_scramble([1, 2, 3, 4, 5], 0, 2), np.array([1, 3, 2, 4, 5]))), False)
        np.testing.assert_equal(all(np.equal(ga.mutation_scramble([1, 2, 3, 4, 5, 6], 0, 6), np.array([1, 3, 2, 4, 5, 6]))), False)
        np.testing.assert_equal(all(np.equal(ga.mutation_scramble([1, 2, 3, 4, 5, 6, 7], 3, 5), np.array([1, 3, 2, 4, 5, 6, 7]))), False)

    def test_inversion(self):
        np.testing.assert_array_equal(ga.mutation_inversion([1, 2, 3, 4, 5], 0, 2), np.array([3, 2, 1, 4, 5]))
        np.testing.assert_array_equal(ga.mutation_inversion([1, 2, 3, 4, 5, 6], 2, 4), np.array([1, 2, 5, 4, 3, 6]))
        np.testing.assert_array_equal(ga.mutation_inversion([1, 2, 3, 4, 5, 6, 7], 5, 6), np.array([1, 2, 3, 4, 5, 7, 6]))


if __name__ == "__main__":
    unittest.main()
