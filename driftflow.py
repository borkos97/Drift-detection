import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection import PageHinkley, DDM

PATH = './weatherAUS.csv'


def plots(stream, detected_change, name, beginning_stream, end_tables):
    plt.figure(figsize=(24, 10))
    plt.plot(detected_change[:end_tables], color='red', marker='o', markersize="8", label='Anomaly')
    plt.plot(stream[beginning_stream:end_tables], color="green", linestyle='-', label='Time series')
    red_patch = mpatches.Patch(color='red', label='Anomaly')
    green_lines = mpatches.Patch(color='green', label='Time series')
    plt.legend(handles=[green_lines, red_patch])
    plt.title(f'Drift function for the algorithm {name}')
    plt.ylabel('Temperature')
    plt.xlabel('Time series')
    plt.grid(True)
    plt.show()


def prepare_data(path):
    data = pd.read_csv(path, header=0, sep=',', parse_dates=['Date'])
    nr_cols_float = list(data.select_dtypes(include='number').columns)
    nr_cols_string = list(data.select_dtypes(include='object').columns)
    data[nr_cols_float] = data[nr_cols_float].fillna(data[nr_cols_float].mean())
    data.dropna(axis='index', inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).sub(pd.Timestamp('2008-12-01')).dt.days
    data = pd.get_dummies(data, columns=nr_cols_string)
    return data


def make_stream(path):
    data = prepare_data(path)
    stream = DataStream(data, y=None, target_idx=-1, n_targets=1, cat_features=None, name=None, allow_nan=False)
    stream = stream.y
    return stream


def drift_flow(stream, method, name, beginning_stream, end_tables):
    detected_change = []
    detected_warning = []
    number_of_changes = 0
    for i in range(len(stream)):
        method.add_element(stream[i])
        if method.detected_warning_zone():
            print(f'Warning zone has been detected in data: {stream[i]} - of index: {i}')
            detected_warning.append((stream[i]))
        if method.detected_change():
            detected_change.append(stream[i])
            print(f'Change has been detected in data: {stream[i]} - of index: {i}')
            number_of_changes += 1
        else:
            detected_change.append(None)
    print(f'{name} Detected changes: {number_of_changes}')
    print(f'{name} Detected warning zones: {str(len(detected_warning))}')
    plots(stream, detected_change, name, beginning_stream, end_tables)


stream = make_stream(PATH)

drift_flow(stream, EDDM(), 'EDDM', 0, 500)
drift_flow(stream, HDDM_A(), 'HDDM_A', 0, 500)
drift_flow(stream, HDDM_W(), 'HDDM_W', 0, 500)
drift_flow(stream, PageHinkley(), 'PH', 0, 500)
drift_flow(stream, DDM(), 'DDM', 0, 500)
