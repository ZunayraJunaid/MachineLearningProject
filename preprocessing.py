# preprocessing.py

import numpy as np
import pickle

def preprocess_input(form_data):
    binary_map = {
        'normal': 0, 'abnormal': 1,
        'present': 1, 'notpresent': 0,
        'yes': 1, 'no': 0,
        'good': 1, 'poor': 0
    }

    features = [
        float(form_data.get('age', 0)),
        float(form_data.get('bp', 0)),
        float(form_data.get('sg', 0)),
        float(form_data.get('al', 0)),
        float(form_data.get('su', 0)),
        binary_map.get(form_data.get('rbc', 'normal'), 0),
        binary_map.get(form_data.get('pc', 'normal'), 0),
        binary_map.get(form_data.get('pcc', 'notpresent'), 0),
        binary_map.get(form_data.get('ba', 'notpresent'), 0),
        float(form_data.get('bgr', 0)),
        float(form_data.get('bu', 0)),
        float(form_data.get('sc', 0)),
        float(form_data.get('sod', 0)),
        float(form_data.get('pot', 0)),
        float(form_data.get('hemo', 0)),
        float(form_data.get('pcv', 0)),
        float(form_data.get('wbcc', 0)),
        float(form_data.get('rbcc', 0)),
        binary_map.get(form_data.get('htn', 'no'), 0),
        binary_map.get(form_data.get('dm', 'no'), 0),
        binary_map.get(form_data.get('cad', 'no'), 0),
        binary_map.get(form_data.get('appet', 'good'), 0),
        binary_map.get(form_data.get('pe', 'no'), 0),
        binary_map.get(form_data.get('ane', 'no'), 0),
    ]

    with open('preprocessing/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return input_scaled