import pandas as pd
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection import HDDM_A

def prepare_data(filename):
    data = pd.read_csv(filename, header=0, sep=',', parse_dates=['Date'])
    nr_cols_float = list(data.select_dtypes(include='number').columns)
    data[nr_cols_float] = data[nr_cols_float].fillna(data[nr_cols_float].mean())
    data.dropna(axis='index', inplace=True)
    return data


def check_nulls(data):
    print(data.isnull().sum())

def eddm(data):
    pass

def hddm_a(data):
    pass


data = prepare_data('weatherAUS.csv')
# check_nulls(data)
hddm_a(data)
pass
