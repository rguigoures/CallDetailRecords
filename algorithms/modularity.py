import networkx as nx
from utils.data_processing import *


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


if __name__ == '__main__':
    cdr_data = process_data('data/mobile-phone-activity/sms-call-internet-mi-2013-11-01.csv')
    print 'loading data'
    g = cdr_graph(cdr_data)
    print 'loading graph'
    clusters, clustering_modularity = g.partition_graph(g.graph)

