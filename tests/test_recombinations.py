import unittest

import numpy as np

from genetic_nqueens import recombination


class TestRecombinations(unittest.TestCase):

    def test_pmx(self):
        new_individual1, new_individual2 = recombination.pmx(individual1=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                                             individual2=np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]),
                                                             gene1=3, gene2=6)
        np.testing.assert_array_equal(new_individual1, np.array([9, 3, 2, 4, 5, 6, 7, 1, 8]))
        np.testing.assert_array_equal(new_individual2, np.array([1, 7, 3, 8, 2, 6, 5, 4, 9]))

    def test_edge(self):
        new_individual1 = recombination.edge(individual1=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                             individual2=np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]),
                                             current_element=1)
        np.testing.assert_array_equal(np.array_equal(new_individual1, np.array([1, 5, 6, 2, 8, 7, 3, 9, 4]))
                                      or np.array_equal(new_individual1, np.array([1, 5, 6, 7, 8, 2, 3, 9, 4]))
                                      or np.array_equal(new_individual1, np.array([1, 5, 6, 2, 8, 7, 3, 4, 9]))
                                      or np.array_equal(new_individual1, np.array([1, 5, 6, 7, 8, 2, 3, 4, 9])), True)

    def test_order(self):
        new_individual1, new_individual2 = recombination.order(individual1=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                                               individual2=np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]),
                                                               gene1=3, gene2=6)
        np.testing.assert_array_equal(new_individual1, np.array([3, 8, 2, 4, 5, 6, 7, 1, 9]))
        np.testing.assert_array_equal(new_individual2, np.array([3, 4, 7, 8, 2, 6, 5, 9, 1]))

    def test_cycle(self):
        new_individual1, new_individual2 = recombination.cycle(individual1=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                                               individual2=np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]),
                                                               gene1=3, gene2=6)
        np.testing.assert_array_equal(new_individual1, np.array([1, 3, 7, 4, 2, 6, 5, 8, 9]))
        np.testing.assert_array_equal(new_individual2, np.array([9, 2, 3, 8, 5, 6, 7, 1, 4]))


if __name__ == "__main__":
    unittest.main()
