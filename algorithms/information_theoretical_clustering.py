from random import shuffle
from utils.data_processing import *


class InformationTheoreticalClustering(object):

    def __init__(self, cdr_dataframe):
        self.cdr_dataframe = cdr_dataframe
        self.adjacency_matrix = self.__build_adjacency_matrix()

    def __build_adjacency_matrix(self):
        """
        transforms the cdr data frame into an adjacency matrix

        :return: adjacency matrix (numpy format)
        """
        cdr_bipartite_matrix = convert_dataframe_to_matrix(self.cdr_dataframe, row='CellID', column='countrycode')
        cdr_adjacency_matrix = project_matrix(cdr_bipartite_matrix, axis='CellID', null_diagonal=True)
        return cdr_adjacency_matrix.as_matrix()

    @staticmethod
    def random_partition(L, k):
        """
        Generate k random balanced partitions of the list L

        :param L: list to be partitioned
        :param k: int number of partitions
        :return: a list of list
        """
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
    def build_cluster_adjacency_matrix(M, clusters):
        """
        group rows and columns of a matrix as specified in the clusters parameters

        :param M: adjacency matrix size n
        :param clusters: list of k (number of clusters) list of indices
        :return: an adjacency matrix of size k
        """
        row_cluster_adjacency_matrix = np.zeros((len(clusters), M.shape[1]))
        for i, cluster in enumerate(clusters):
            row_cluster_adjacency_matrix[i, :] = M[cluster, :].sum(axis=0)
        cluster_adjacency_matrix = np.zeros((len(clusters), len(clusters)))
        for i, cluster in enumerate(clusters):
            cluster_adjacency_matrix[:, i] = row_cluster_adjacency_matrix[:, cluster].sum(axis=1)
        return cluster_adjacency_matrix

    def information_theoretical_clustering(self, k):
        """
        algorithm finding the best partition into k clusters

        :param k: number of clusters
        :return: clusters and the corresponding adjacency matrix
        """
        clusters = self.random_partition([i for i in range(self.adjacency_matrix.shape[0])], k)
        cluster_adjacency_matrix = self.build_cluster_adjacency_matrix(self.adjacency_matrix, clusters)
        clustering_mi = np.nansum(self.compute_mutual_information(cluster_adjacency_matrix))
        improvement = 1
        while improvement > 0.:
            for cluster in clusters:
                for i in range(len(cluster)):
                    node = cluster.pop(0)
                    best_mi = 0
                    for j in range(k):
                        clusters[j].append(node)
                        cluster_adjacency_matrix = self.build_cluster_adjacency_matrix(self.adjacency_matrix, clusters)
                        mi = np.nansum(self.compute_mutual_information(cluster_adjacency_matrix))
                        if best_mi < mi:
                            best_mi = mi
                            best_cluster = j
                        del clusters[j][-1]
                    clusters[best_cluster].append(node)
            cluster_adjacency_matrix = self.build_cluster_adjacency_matrix(self.adjacency_matrix, clusters)
            improvement = np.nansum(self.compute_mutual_information(cluster_adjacency_matrix)) - clustering_mi
            clustering_mi = np.nansum(self.compute_mutual_information(cluster_adjacency_matrix))
        return clusters, cluster_adjacency_matrix

if __name__ == '__main__':
    cdr_data = process_data('../data/mobile-phone-activity/sms-call-internet-mi-2013-11-01.csv')
    print 'loading data'
    g = InformationTheoreticalClustering(cdr_data)
    print 'loading graph'
    g.information_theoretical_clustering(k=10)