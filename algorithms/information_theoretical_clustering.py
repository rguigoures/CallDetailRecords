import numpy as np
from scipy.stats import entropy
from utils.data_processing import *
from random import shuffle


class InformationTheoreticalClustering(object):

    def __init__(self, cdr_dataframe):
        self.cdr_dataframe = cdr_dataframe
        self.adjacency_matrix, self.cell_index, self.country_index = self.__build_adjacency_matrix()
        self.js_matrix = {'cell': np.ones((self.adjacency_matrix.shape[0], self.adjacency_matrix.shape[0])) * np.inf,
                          'country': np.ones((self.adjacency_matrix.shape[1], self.adjacency_matrix.shape[1])) * np.inf}

    def __build_adjacency_matrix(self):
        """
        transforms the cdr data frame into an adjacency matrix

        :return: adjacency matrix (numpy format)
        """
        cdr_bipartite_matrix = convert_dataframe_to_matrix(self.cdr_dataframe, row='CellID', column='countrycode')
        adjacency_matrix = cdr_bipartite_matrix.as_matrix()
        non_zero_sms_countries = list(np.where(adjacency_matrix.sum(axis=0) != 0)[0])
        adjacency_matrix = adjacency_matrix[:, non_zero_sms_countries]
        cell_index = cdr_bipartite_matrix.index.values.tolist()
        country_index = [ind for i, ind in enumerate(cdr_bipartite_matrix.columns.values.tolist())
                         if i in non_zero_sms_countries]
        return adjacency_matrix, cell_index, country_index

    @staticmethod
    def random_partition(L, k, shuffle_list=True):
        """
        Generate k random balanced partitions of the list L

        :param L: list to be partitioned
        :param k: int number of partitions
        :return: a list of list
        """
        if shuffle_list:
            shuffle(L)
        partition = []
        chunk_sizes = np.random.multinomial(len(L), [1 / float(k)] * k)
        for chunk_size in chunk_sizes:
            partition.append(L[:chunk_size])
            del L[:chunk_size]
        return partition

    @staticmethod
    def compute_mutual_information(matrix):
        """
        compute the mutual information matrix

        :param matrix: cooccurence matrix
        :return: mutual information matrix
        """
        normalized_matrix = matrix / float(matrix.sum())
        mi_matrix = normalized_matrix * (np.log(normalized_matrix)
                                         - np.log(normalized_matrix.sum(axis=0))
                                         - np.log(normalized_matrix.sum(axis=1)[np.newaxis].T))
        return mi_matrix

    @staticmethod
    def build_cluster_join_probability_matrix(matrix, clusters, dimension):
        """
        group rows and columns of a matrix as specified in the clusters parameters

        :param M: adjacency matrix size n
        :param clusters: list of k (number of clusters) list of indices
        :return: an adjacency matrix of size k
        """
        M_x_and_cy = np.zeros((matrix.shape[0], len(clusters)))
        M_cx_and_y = np.zeros((len(clusters), matrix.shape[1]))
        for i, cluster in enumerate(clusters):
            if dimension == 'cell':
                M_cx_and_y[i, :] = matrix[cluster, :].sum(axis=0) / float(len(cluster))
                mat = M_cx_and_y / M_cx_and_y.sum()
            elif dimension == 'country':
                M_x_and_cy[:, i] = matrix[:, cluster].sum(axis=1)[np.newaxis].T.ravel() / float(len(cluster))
                mat = M_x_and_cy / M_x_and_cy.sum()
        return mat

    @staticmethod
    def build_partionned_probability_matrix(data_matrix, cluster_matrix, cell_clusters, country_clusters):
        """
        group rows and columns of a matrix as specified in the clusters parameters

        :param data_matrix: adjacency matrix size nxm
        :param cluster_matrix: cluster adjacency matrix size kxl
        :param cell_clusters: list of k (number of clusters) list of indices
        :param country_clusters: list of k (number of countries) list of indices
        :return: an adjacency matrix of size k
        """
        mat = np.zeros(data_matrix.shape)
        for i, cell_cluster in enumerate(cell_clusters):
            for j, country_cluster in enumerate(country_clusters):
                mat[np.ix_(cell_cluster, country_cluster)] = cluster_matrix[i,j]
        mat /= mat.sum()
        return mat

    def coclustering(self, k, l):
        """
        performs the clustering algorithm

        :param k: number of clusters of antennas
        :param l: number of clusters of countries
        :return: clusters of antennas and clusters of countries
        """
        stop_iterate = 0
        n_obs = self.adjacency_matrix.sum()
        P_x_and_y = self.adjacency_matrix / float(n_obs)
        P_y_given_x = P_x_and_y / P_x_and_y.sum(axis=1)[np.newaxis].T
        P_x_given_y = P_x_and_y / P_x_and_y.sum(axis=0)
        cell_clusters = self.random_partition(list(range(self.adjacency_matrix.shape[0])), k)
        country_clusters = self.random_partition(list(range(self.adjacency_matrix.shape[1])), l)
        partitionned_matrix = self.adjacency_matrix
        best_mi = 0
        while stop_iterate < 5:
            cell_kl, country_kl = {}, {}
            new_cell_clusters = [[] for _ in range(k)]
            new_country_clusters = [[] for _ in range(l)]
            for i in range(P_y_given_x.shape[0]):
                cell_kl[i] = {}
                for j in range(k):
                    P_y_and_cx = self.build_cluster_join_probability_matrix(partitionned_matrix,
                                                                            cell_clusters, 'cell')
                    P_y_given_cx = P_y_and_cx / P_y_and_cx.sum(axis=1)[np.newaxis].T
                    cell_kl[i][j] = entropy(P_y_given_x[i], P_y_given_cx[j])
                j = min(cell_kl[i].iteritems(), key=operator.itemgetter(1))[0]
                new_cell_clusters[j].append(i)
            for i in range(P_x_given_y.shape[1]):
                country_kl[i] = {}
                for j in range(l):
                    P_x_and_cy = self.build_cluster_join_probability_matrix(partitionned_matrix,
                                                                            country_clusters, 'country')
                    P_x_given_cy = P_x_and_cy / P_x_and_cy.sum(axis=0)
                    country_kl[i][j] = entropy(P_x_given_y.T[i], P_x_given_cy.T[j])
                j = min(country_kl[i].iteritems(), key=operator.itemgetter(1))[0]
                new_country_clusters[j].append(i)
            if cell_clusters == new_cell_clusters and country_clusters == new_country_clusters:
                break
            cell_clusters = new_cell_clusters
            country_clusters = new_country_clusters
            cluster_matrix = self.build_cluster_join_probability_matrix(self.adjacency_matrix, cell_clusters,
                                                                        dimension='cell')
            cluster_matrix = self.build_cluster_join_probability_matrix(cluster_matrix, country_clusters,
                                                                        dimension='country')
            partitionned_matrix = self.build_partionned_probability_matrix(self.adjacency_matrix, cluster_matrix,
                                                     cell_clusters, country_clusters)
            if best_mi < np.nansum(self.compute_mutual_information(cluster_matrix)):
                best_mi = np.nansum(self.compute_mutual_information(cluster_matrix))
                best_cell_clusters, best_country_clusters = cell_clusters, country_clusters
                stop_iterate = 0
            else:
                stop_iterate += 1
        return best_cell_clusters, best_country_clusters