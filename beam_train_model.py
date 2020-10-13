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
from tensorflow.keras.models import load_model, save_model
import tensorflow.keras.backend as K

from utils import set_seed, OneCycleLR
from resnet import conv_block, residual_body, stem
from beam_utils import load_data, get_beams_output

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

LIDAR_TRAIN_FILE = LIDAR_DATA_DIR / 'lidar_train.npz'
LIDAR_VAL_FILE = LIDAR_DATA_DIR / 'lidar_validation.npz'

COORDINATES_TRAIN_FILE = COORDINATES_DATA_DIR / 'my_coord_train.npz'
COORDINATES_VAL_FILE = COORDINATES_DATA_DIR / 'my_coord_validation.npz'

BEAMS_TRAIN_FILE = BEAMS_DATA_DIR / 'beams_output_train.npz'
BEAMS_VAL_FILE = BEAMS_DATA_DIR / 'beams_output_validation.npz'

BATCH_SIZE = 32
BEST_WEIGTHS = 'my_model_weights.h5'

KERNEL_REG = 1.e-4

# set random seeds and return numpy random generator:
set_seed(123)

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


X_lidar_train = load_data(LIDAR_TRAIN_FILE, 'input')
X_lidar_val = load_data(LIDAR_VAL_FILE, 'input')

X_coord_train = load_data(COORDINATES_TRAIN_FILE, 'coordinates')
X_coord_val = load_data(COORDINATES_VAL_FILE, 'coordinates')

X_coord_train, coord_means, coord_stds = process_coordinates(X_coord_train)

np.savez('coord_train_stats.npz', coord_means=coord_means, coord_stds=coord_stds)

X_coord_val, _, _ = process_coordinates(X_coord_val, coord_means, coord_stds)

Y_train, num_classes = get_beams_output(BEAMS_TRAIN_FILE)
Y_val, _ = get_beams_output(BEAMS_VAL_FILE)

print("Shape of lidar data:", X_lidar_train.shape)
print("Shape of coordinate data:", X_coord_train.shape)
print("Number of classes", num_classes)
print("Shape of ground truth:", Y_train.shape)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x1, x2, y, batch_size, shuffle=False):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x1))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = (self.x1[indexes], self.x2[indexes])
        Y = self.y[indexes]

        return X, Y


train_generator = DataGenerator(X_lidar_train, X_coord_train, Y_train, BATCH_SIZE, shuffle=True)
val_generator = DataGenerator(X_lidar_val, X_coord_val, Y_val, BATCH_SIZE)

# ## Model

params = { "kernel_regularizer": regularizers.l2(KERNEL_REG) }

def create_model_lidar(inp):
    y = stem([32, 32, 64], strides=(1, 2), **params)(inp)
    y = residual_body(64, [2, 2], [2, 2], **params)(y)

    y = conv_block(8, kernel_size=1, **params)(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.25)(y)
    y = layers.Dense(256, activation='relu', **params)(y)

    return y


def create_model_coord(inp):
    y = layers.Dense(8, activation='relu', **params)(inp)
    y = layers.Dense(16, activation='relu', **params)(y)
    y = layers.Dense(64, activation='relu', **params)(y)
    y = layers.Dense(256, activation='relu', **params)(y)

    return y


def create_model(input_shape_lidar, input_shape_coord, classes):
    inp_lidar = keras.Input(shape=input_shape_lidar)
    inp_coord = keras.Input(shape=input_shape_coord)

    out_lidar = create_model_lidar(inp_lidar)
    out_coord = create_model_coord(inp_coord)

    y = layers.Concatenate()([out_lidar, out_coord])
    y = layers.Dropout(0.5)(y)
    out = layers.Dense(classes, activation='softmax', **params)(y)

    model = keras.Model(inputs = [inp_lidar, inp_coord], outputs = out)
    return model


model = create_model(X_lidar_train.shape[1:], X_coord_train.shape[1:], num_classes)

model.summary(line_length=128)

# ## Training

K.clear_session()

optim = optimizers.Adam()

model_metrics = [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy()]

model_loss = losses.CategoricalCrossentropy()

model.compile(loss = model_loss, optimizer = optim, metrics = model_metrics)

# lrfinder = LearningRateFinder(model)

# lrfinder.find(train_generator, 1e-6, 1e0, 1)
# lrfinder.plot_loss()

# K.set_value(model.optimizer.lr, 1e-2)

EPOCHS = 50

one_cycle_sheduler = OneCycleLR(max_lr=1e-2, total_steps = EPOCHS * len(train_generator))

checkpoint = callbacks.ModelCheckpoint(BEST_WEIGTHS, monitor='val_top_k_categorical_accuracy',
                                       verbose=1, save_best_only=True,
                                       save_weights_only=True, mode='max', save_freq='epoch')

tb_log = callbacks.TensorBoard(log_dir='./logs')

callbacks = [one_cycle_sheduler, checkpoint]

hist = model.fit(train_generator,
                 validation_data=val_generator,
                 epochs=EPOCHS,
                 callbacks=callbacks,
                 verbose=1)

#model.save_weights(BEST_WEIGTHS, save_format='h5')

# ## Evaluation

model.load_weights(BEST_WEIGTHS)

model.evaluate(val_generator, verbose=1)

model_json = model.to_json()
with open('my_model.json', "w") as json_file:
    json_file.write(model_json)
