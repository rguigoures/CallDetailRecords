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

    def test_build_cluster_join_probability_matrix(self):
        matrix = np.array([[1, 1, 1], [1, 1, 1], [2, 0, 2], [2, 0, 2]])
        dimension = 'cell'
        clusters = [[0, 1], [2, 3]]
        expected_mat = [[1/7., 1/7., 1/7.], [2/7., 0, 2/7.]]
        mat = InformationTheoreticalClustering.build_cluster_join_probability_matrix(matrix, clusters, dimension)
        self.assertEqual(np.array_equal(mat, expected_mat), True)

    def test_build_partionned_probability_matrix(self):
        data_matrix = np.array([[1, 1, 1], [1, 1, 1], [2, 0, 2], [2, 0, 2]])
        cluster_matrix = np.array([[4, 2], [8, 0]])
        cell_clusters = [[0, 1], [2, 3]]
        country_clusters = [[0, 2], [1]]
        expected_mat = np.array([[1/13., 1/26., 1/13.], [1/13., 1/26., 1/13.], [2/13., 0, 2/13.], [2/13., 0, 2/13.]])
        mat = InformationTheoreticalClustering.build_partionned_probability_matrix(data_matrix, cluster_matrix,
                                                                                   cell_clusters, country_clusters)
        self.assertEqual(np.array_equal(mat, expected_mat), True)

if __name__ == '__main__':
    unittest.main()
