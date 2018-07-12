import pandas
import numpy as np
from matplotlib import pyplot, colors
import json
import operator


def get_antennas_with_k_highest_count(df, k):
    """
    returns the k antennas with highest call counts

    :param df: pandas dataframe (calls, CellId, count)
    :param k: number of top antennas to be returned
    :return: list of th top k antennas
    """
    agg_df = df.groupby(['CellID'])['callout'].agg('sum').to_frame()
    counts = agg_df.to_dict(orient='dict')
    return [p[0] for p in sorted(counts['callout'].iteritems(), key=operator.itemgetter(1))[-k:]]

def get_countries_with_k_highest_count(df, k):
    """
    returns the k countries with highest call counts

    :param df: pandas dataframe (calls, CellId, count)
    :param k: number of top antennas to be returned
    :return: list of th top k countries
    """
    agg_df = df.groupby(['countrycode'])['callout'].agg('sum').to_frame()
    counts = agg_df.to_dict(orient='dict')
    return [p[0] for p in sorted(counts['callout'].iteritems(), key=operator.itemgetter(1))[-k:]]

def process_data(file_name, truncate=0):
    """
    parse a csv file, build a pandas dataframe, and aggregate the data

    :param file_name: path to a csv file
    :param truncate: number of antennas to use for the analysis (top k antennas)
    :return: aggregated pandas dataframe, antennas indices, countries indices
    """
    data = pandas.read_csv(file_name).fillna(0)
    data = data[data.countrycode != 0]
    data = data[data.countrycode != 39]
    data = data[data.countrycode < 1000]
    most_called_countries = get_countries_with_k_highest_count(data, 100)
    most_used_antennas = get_antennas_with_k_highest_count(data, truncate)
    data = data.loc[data['CellID'].isin(most_used_antennas)]
    data = data.loc[data['countrycode'].isin(most_called_countries)]
    aggregated_data = data.groupby(['CellID', 'countrycode'])['callout'].agg('sum')
    aggregated_data = aggregated_data.to_frame()
    aggregated_data.reset_index(level=aggregated_data.index.names, inplace=True)
    aggregated_data.callout *= 100000
    aggregated_data.callout = aggregated_data.callout.astype(int)
    return aggregated_data, most_used_antennas, most_called_countries


def convert_dataframe_to_matrix(dataframe, row, column):
    """
    Builds a matrix from an aggregated data frame

    :param dataframe: aggregated data frame
    :param row: variable to be in row
    :param column: variable to be in column
    :return: cooccurrence matrix
    """
    matrix = dataframe.pivot_table(index=row, columns=column).fillna(0).astype('int')
    matrix.columns = pandas.Index(sorted(list(set(dataframe[column]))), dtype='object', name=column)
    return matrix


def project_matrix(matrix, axis, null_diagonal=True):
    """
    performs matrix projection: P=M.TM

    :param matrix: input matrix
    :param axis: axis along which the ajacency matrix should be projected
    :param null_diagonal: set the diagonal to 0
    :return: square matrix which size is the projected axis size
    """
    if matrix.index.name == axis:
        proj_matrix = matrix.dot(matrix.transpose())
    elif matrix.column.name == axis:
        proj_matrix = matrix.transpose().dot(matrix)
    if null_diagonal:
        np.fill_diagonal(proj_matrix.values, 0)
    return proj_matrix


def get_color_map(k):
    """
    yields color for each of the k clusters

    :param k: int
    :return: hexa color code
    """
    cmap = pyplot.cm.get_cmap('Set3', k)
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        yield colors.rgb2hex(rgb)


def plot_clusters(clusters, antenna_mapping):
    """
    plots clusters on a map and generates a geojson file

    :param clusters: list of clusters of antennas
    :param antenna_mapping: mapping the antenna index with the antenna identifier
    """
    antennas_location = open('data/mobile-phone-activity/milano-grid.geojson')
    antennas_location = json.load(antennas_location)
    clustered_antennas_location = {"type": antennas_location["type"],
                                   "features": []}
    colors = get_color_map(len(clusters))
    for i, cluster in enumerate(clusters):
        color = next(colors)
        print i, color
        for antenna in cluster:
            clustered_antennas_location['features'].append(antennas_location['features'][antenna_mapping[antenna]-1])
            clustered_antennas_location['features'][-1]['properties']['stroke'] = color
            clustered_antennas_location['features'][-1]['properties']['fill'] = color
            clustered_antennas_location['features'][-1]['properties']['stroke-opacity'] = .8
            clustered_antennas_location['features'][-1]['properties']['fill-opacity'] = .8
    with open('antenna.geojson', 'w') as outfile:
        json.dump(clustered_antennas_location, outfile)