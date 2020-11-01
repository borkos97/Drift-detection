import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection import PageHinkley, DDM
from skmultiflow.drift_detection.hddm_w import HDDM_W


def prepare_data(filename):
    data = pd.read_csv(filename, header=0, sep=',', parse_dates=['Date'])
    nr_cols_float = list(data.select_dtypes(include='number').columns)
    nr_cols_string = list(data.select_dtypes(include='object').columns)
    data[nr_cols_float] = data[nr_cols_float].fillna(data[nr_cols_float].mean())
    data.dropna(axis='index', inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).sub(pd.Timestamp('2008-12-01')).dt.days
    data = pd.get_dummies(data, columns=nr_cols_string)
    return data


def check_nulls(data):
    print(data.isnull().sum())


def make_stream(data):
    stream = DataStream(data, y=None, target_idx=-1, n_targets=1, cat_features=None, name=None, allow_nan=False)
    stream = stream.y
    return stream


def eddm(stream):
    detected_change = []
    eddm = EDDM()
    data_stream = stream
    for i in range(len(stream)):
        eddm.add_element(data_stream[i])
        if eddm.detected_change():
            detected_change.append((data_stream[i]))
            print("Change has been detected in data: {}"
                  " - of index: {}".format(data_stream[i], i))
    print("EDDM Detected changes: " + str(len(detected_change)))


def hddm_a(stream):
    detected_change = []

    hddm_a = HDDM_A()
    data_stream = stream
    for i in range(len(stream)):
        hddm_a.add_element(data_stream[i])

        if hddm_a.detected_change():
            detected_change.append((data_stream[i]))
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    print("HDDM_A Detected changes: " + str(len(detected_change)))


def ph(stream):
    detected_change = []
    ph = PageHinkley()
    data_stream = stream
    for i in range(len(stream)):
        ph.add_element(data_stream[i])
        if ph.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_change.append((data_stream[i]))
    print("PH Detected changes: " + str(len(detected_change)))


def hddm_w(stream):
    detected_change = []

    hddm_w = HDDM_W()
    data_stream = stream
    for i in range(len(stream)):
        hddm_w.add_element(data_stream[i])

        if hddm_w.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_change.append((data_stream[i]))
    print("HDDM_W Detected changes: " + str(len(detected_change)))


def ddm(stream):
    detected_change = []

    ddm = DDM()
    data_stream = stream
    for i in range(len(stream)):
        ddm.add_element(data_stream[i])

        if ddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_change.append((data_stream[i]))
    print("DDM Detected changes: " + str(len(detected_change)))


data = prepare_data('weatherAUS.csv')
# check_nulls(data)
stream = make_stream(data)
eddm(stream)
# hddm_a(stream)
# ph(stream)
# hddm_w(stream)
# ddm(stream)
