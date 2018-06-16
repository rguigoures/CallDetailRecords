import unittest
import pandas
import data_processing

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.df = pandas.DataFrame(data=[['R1', 'C1', 1],
                                         ['R1', 'C3', 3],
                                         ['R2', 'C1', 2],
                                         ['R2', 'C2', 2]],
                                   columns=['row', 'column', 'counts'])

    def test_convert_to_matrix(self):
        matrix = data_processing.convert_dataframe_to_matrix(self.df, row='row', column='column')
        expected_matrix = pandas.DataFrame(data=[[1, 0, 3], [2, 2, 0]],
                                           index=['R1', 'R2'],
                                           columns=['C1', 'C2', 'C3'])
        expected_matrix.index.name = 'row'
        self.assertEqual(matrix.equals(expected_matrix), True)

    def test_project_matrix(self):
        matrix = data_processing.convert_dataframe_to_matrix(self.df, row='row', column='column')
        proj_matrix = data_processing.project_matrix(matrix, axis='row')
        expected_proj_matrix = pandas.DataFrame(data=[[0, 2],
                                                      [2, 0]],
                                                index=['R1', 'R2'],
                                                columns=['R1', 'R2'])
        self.assertEqual(proj_matrix.equals(expected_proj_matrix), True)


if __name__ == '__main__':
    unittest.main()
