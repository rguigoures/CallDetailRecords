import unittest
from algorithms.information_theoretical_clustering import InformationTheoreticalClustering
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_random_partition(self):
        L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        partition = InformationTheoreticalClustering.random_partition(L, 3)
        self.assertEqual(len(partition), 3)
        for p in partition:
            self.assertGreater(len(p), 0)

    def test_mutual_information(self):
        A = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        mi = np.nansum(InformationTheoreticalClustering.compute_mutual_information(A))
        expected_mi = np.log(3)
        self.assertAlmostEqual(mi, expected_mi)
        A = np.array([[1, 1], [1, 1]])
        mi = np.nansum(InformationTheoreticalClustering.compute_mutual_information(A))
        expected_mi = 0
        self.assertAlmostEqual(mi, expected_mi)

    def test_cluster_matrix(self):
        A = np.array([[10, 1, 0], [0, 12, 0], [0, 0, 10]])
        B = InformationTheoreticalClustering.build_cluster_adjacency_matrix(A, [[0, 1], [2]])
        expected_B = np.array([[23, 0], [0, 10]])
        self.assertEqual(np.array_equal(B, expected_B), True)

if __name__ == '__main__':
    unittest.main()
