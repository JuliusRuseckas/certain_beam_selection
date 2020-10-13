#!/usr/bin/env python3
# coding: utf-8

# # Beam prediction

# ## Configuration

# Imports

import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, optimizers, losses, callbacks, regularizers
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K

from resnet import AddRelu
from beam_utils import load_data

# Argument parsing

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('data_folder', help='Location of the data directory', type=str)
args = parser.parse_args()

# Configuration

DATA_DIR = Path(args.data_folder) / 'baseline_data'

LIDAR_DATA_DIR = DATA_DIR / 'lidar_input'
COORDINATES_DATA_DIR = DATA_DIR / 'coord_input'
IMAGES_DATA_DIR = DATA_DIR / 'image_input'
BEAMS_DATA_DIR = DATA_DIR / 'beam_output'

LIDAR_TEST_FILE = LIDAR_DATA_DIR / 'lidar_test.npz'

COORDINATES_TEST_FILE = COORDINATES_DATA_DIR / 'my_coord_test.npz'

BATCH_SIZE = 32
BEST_WEIGTHS = 'my_model_weights.h5'

KERNEL_REG = 1.e-4


num_classes = 256

# ## Data generator

def normalize_data(X, means=None, stds=None):
    if means is None: means = np.mean(X, axis=0)
    if stds is None: stds = np.std(X, axis=0)
    X_norm = (X - means) / stds
    return X_norm, means, stds


def process_coordinates(X, means=None, stds=None):
    X_xyz, means, stds = normalize_data(X[:, :3], means, stds)
    X = np.concatenate((X_xyz, X[:, 3:]), axis=1)
    return X, means, stds


X_lidar_test = load_data(LIDAR_TEST_FILE, 'input')

X_coord_test = load_data(COORDINATES_TEST_FILE, 'coordinates')

cache = np.load('coord_train_stats.npz')
coord_means = cache['coord_means']
coord_stds = cache['coord_stds']

X_coord_test, _, _ = process_coordinates(X_coord_test, coord_means, coord_stds)

print("Shape of lidar data:", X_lidar_test.shape)
print("Shape of coordinate data:", X_coord_test.shape)
print("Number of classes", num_classes)


class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, x1, x2, batch_size, shuffle=False):
        self.x1 = x1
        self.x2 = x2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x1))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))
    
    def on_epoch_end(self):
        if self.shuffle:
            rg.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = (self.x1[indexes], self.x2[indexes])
        return (X,)


test_generator = TestDataGenerator(X_lidar_test, X_coord_test, BATCH_SIZE)

# ## Model

with open('my_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json, custom_objects={'AddRelu': AddRelu})

optim = optimizers.Adam()

model_metrics = [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy()]

model_loss = losses.CategoricalCrossentropy()

model.compile(loss = model_loss, optimizer = optim, metrics = model_metrics)

model.load_weights(BEST_WEIGTHS)

# ## Testing

preds = model.predict(test_generator, verbose=1)

np.savetxt('beam_test_pred.csv', preds, delimiter=',')
