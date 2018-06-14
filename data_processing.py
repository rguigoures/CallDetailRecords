import pandas
import pystan

def process_data(file_name):
    data = pandas.read_csv(file_name).fillna(0)
    data = data[data.countrycode != 0]
    aggregated_data = data.groupby(['CellID', 'countrycode'])['smsin'].agg('sum')
    aggregated_data = aggregated_data.to_frame() * 1e4
    aggregated_data.smsin = aggregated_data.smsin.astype(int)
    return aggregated_data

def model():
    pystan.StanModel(file='model.stan')

if __name__ == '__main__':
    #data = process_data('data\mobile-phone-activity\sms-call-internet-mi-2013-11-01.csv')
    #print data.head
    model()