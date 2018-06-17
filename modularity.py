import networkx as nx
from data_processing import *
from random import shuffle


class cdr_graph():

    def __init__(self, cdr_dataframe):
        self.cdr_dataframe = cdr_dataframe
        self.graph = self.__build_graph()

    def __build_graph(self):
        """
        build a networkx graph from a cdr data frame

        :return: networkx graph where nodes are cell ids and edges the number of sms
        """
        cdr_bipartite_matrix = convert_dataframe_to_matrix(self.cdr_dataframe, row='CellID', column='countrycode')
        cdr_adjacency_matrix = project_matrix(cdr_bipartite_matrix, axis='CellID', null_diagonal=True)
        cdr_graph = nx.convert_matrix.from_pandas_adjacency(cdr_adjacency_matrix)
        return max(nx.connected_component_subgraphs(cdr_graph), key=len)

    @staticmethod
    def multigraph_to_weighted_graph(multigraph):
        """
        converts a multigraph to a graph with weights on edges

        :param multigraph: networkx MultiGraph object
        :return: networkx Graph object
        """
        G = nx.Graph()
        for u, v, data in multigraph.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        return G

    def partition_graph(self, graph):
        """
        partition the graph into densely connected clusters

        :param graph: weighted graph
        :param cluster_number: number of clusters to be produced
        :return: partition of nodes, modularity value for the partitioning
        """
        partition = nx.algorithms.community.greedy_modularity_communities(graph, weight='weight')
        return partition, nx.algorithms.community.quality.modularity(graph, partition, weight='weight')

    @staticmethod
    def random_partition(L, k):
        shuffle(L)
        partition = []
        chunk_sizes = np.random.multinomial(len(L), [1 / float(k)] * k)
        for chunk_size in chunk_sizes:
            partition.append(L[:chunk_size])
            del L[:chunk_size]
        return partition

    @staticmethod
    def compute_mutual_information(matrix):
        normalized_matrix = matrix / float(matrix.sum())
        mi_matrix = normalized_matrix * (np.log(normalized_matrix)
                                         - np.log(normalized_matrix.sum(axis=0))
                                         - np.log(normalized_matrix.sum(axis=1)))
        return np.nansum(mi_matrix)

    @staticmethod
    def build_cluster_adjacency_matrix(M, clusters):
        row_cluster_adjacency_matrix = np.zeros((len(clusters), M.shape[1]))
        for i, cluster in enumerate(clusters):
            row_cluster_adjacency_matrix[i, :] = M[cluster, :].sum(axis=0)
        cluster_adjacency_matrix = np.zeros((len(clusters), len(clusters)))
        for i, cluster in enumerate(clusters):
            cluster_adjacency_matrix[:, i] = row_cluster_adjacency_matrix[:, cluster].sum(axis=1)
        return cluster_adjacency_matrix

    def information_theoretical_coclustering(self, graph_matrix, k):
        clusters = self.random_partition([node-1 for node in self.graph.nodes], k)
        cluster_adjacency_matrix = self.build_cluster_adjacency_matrix(graph_matrix, clusters)
        clustering_mi = self.compute_mutual_information(cluster_adjacency_matrix)
        improvement = 1
        while improvement > 0.:
            for cluster in clusters:
                for i in range(len(cluster)):
                    node = cluster.pop(0)
                    best_mi = 0
                    for j in range(k):
                        clusters[j].append(node)
                        cluster_adjacency_matrix = self.build_cluster_adjacency_matrix(graph_matrix, clusters)
                        mi = self.compute_mutual_information(cluster_adjacency_matrix)
                        if best_mi < mi:
                            best_mi = mi
                            best_cluster = j
                        del clusters[j][-1]
                    clusters[best_cluster].append(node)
            cluster_adjacency_matrix = self.build_cluster_adjacency_matrix(graph_matrix, clusters)
            improvement = self.compute_mutual_information(cluster_adjacency_matrix) - clustering_mi
            clustering_mi = self.compute_mutual_information(cluster_adjacency_matrix)
        return clusters, cluster_adjacency_matrix



if __name__ == '__main__':
    cdr_data = process_data('data/mobile-phone-activity/sms-call-internet-mi-2013-11-01.csv')
    print 'loading data'
    g = cdr_graph(cdr_data)
    print 'loading graph'
    '''
    clusters, clustering_modularity = g.partition_graph(g.graph)
    print 'performed 1 partition'
    for cluster in clusters:
        if list(cluster) > 1:
            print cluster
    print clustering_modularity
    '''
    g.information_theoretical_coclustering(nx.to_numpy_matrix(g.graph), k=10)
