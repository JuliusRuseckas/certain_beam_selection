# coding: utf-8

import numpy as np

def load_data(filename, key):
    cache = np.load(filename)
    X = cache[key]
    return X


def beams_remove_small(y, threshold_below_max):
    num = y.shape[0]
    for i in range(num):
        beams = y[i, :]
        logs = 20 * np.log10(beams + 1e-30)
        beams[logs < (np.amax(logs) - threshold_below_max)] = 0
        beams = beams / np.sum(beams)
        y[i, :] = beams
        
    return y


def get_beams_output(filename, threshold_below_max=6):
    y_matrix = load_data(filename, 'output_classification')
    y_matrix = np.abs(y_matrix)
    y_matrix /= np.max(y_matrix) #normalize
    
    num_classes = y_matrix.shape[1] * y_matrix.shape[2]
    y = y_matrix.reshape(y_matrix.shape[0], num_classes)
    
    y = beams_remove_small(y, threshold_below_max)
    return y, num_classes
