import pandas
import numpy as np
#import pystan


def process_data(file_name):
    data = pandas.read_csv(file_name).fillna(0)
    data = data[data.countrycode != 0]
    data = data[data.countrycode != 39]
    #data = data[data.CellID < 500]
    aggregated_data = data.groupby(['CellID', 'countrycode'])['smsin'].agg('sum')
    aggregated_data = aggregated_data.to_frame()
    aggregated_data.reset_index(level=aggregated_data.index.names, inplace=True)
    aggregated_data.smsin *= 100
    aggregated_data.smsin = aggregated_data.smsin.astype(int)
    return aggregated_data


def convert_dataframe_to_matrix(dataframe, row, column):
    matrix = dataframe.pivot_table(index=row, columns=column).fillna(0).astype('int')
    matrix.columns = pandas.Index(sorted(list(set(dataframe[column]))), dtype='object', name=column)
    return matrix


def project_matrix(matrix, axis, null_diagonal=True):
    if matrix.index.name == axis:
        proj_matrix = matrix.dot(matrix.transpose())
    elif matrix.column.name == axis:
        proj_matrix = matrix.transpose().dot(matrix)
    if null_diagonal:
        np.fill_diagonal(proj_matrix.values, 0)
    return proj_matrix


if __name__ == '__main__':
    data = process_data('data/mobile-phone-activity/sms-call-internet-mi-2013-11-01.csv')
    print data.unstack(fill_value=0)