# coding: utf-8

import tempfile
import math
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def show_or_save(fig, filename=None):
    if filename:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()


def show_history(hist, name, ylog=False, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(hist.epoch, hist.history[name], '.-', label="Train " + name)
    ax.plot(hist.epoch, hist.history["val_" + name], '.-', label="Validation " + name)
    if ylog: ax.set_yscale('log')
    ax.set_title(name.capitalize())
    ax.grid()
    ax.legend()
    show_or_save(fig, filename)


def append_history(hist1, hist2):
    for k, v in hist2.history.items():
        hist1.history[k] += v

    last_epoch = hist1.epoch[-1]
    renumbered_epoch = [e + last_epoch + 1 for e in hist2.epoch]
    hist1.epoch += renumbered_epoch

    return hist1


class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        self.reset()

    def reset(self):
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def on_batch_end(self, batch, logs):
        lr = keras.backend.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        stopLoss = self.stopFactor * self.bestLoss

        if self.batchNum > 1 and smooth > stopLoss:
            self.model.stop_training = True
            return

        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        lr *= self.lrMult
        keras.backend.set_value(self.model.optimizer.lr, lr)


    def find(self, train_gen, startLR, endLR, epochs, verbose=1):
        self.reset()

        stepsPerEpoch = len(train_gen)
        numBatchUpdates = epochs * stepsPerEpoch

        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        origLR = keras.backend.get_value(self.model.optimizer.lr)
        keras.backend.set_value(self.model.optimizer.lr, startLR)

        callback = keras.callbacks.LambdaCallback(on_batch_end = lambda batch,
                logs: self.on_batch_end(batch, logs))

        self.model.fit(
            train_gen,
            steps_per_epoch = stepsPerEpoch,
            epochs = epochs,
            verbose = verbose,
            callbacks = [callback])

        self.model.load_weights(self.weightsFile)
        keras.backend.set_value(self.model.optimizer.lr, origLR)


    def plot_loss(self, title = "", filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lrs, self.losses, '.-')
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (Log Scale)")
        ax.set_ylabel("Loss")
        if title != "": ax.set_title(title)
        show_or_save(fig, filename)



class OneCycleLR(keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps, pct_start=0.3, cyclical_momentum=True,
                 max_m=0.95, base_m=0.85, div_factor=25., final_div_factor=1e4):
        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor

        self.cyclical_momentum = cyclical_momentum
        self.base_m = base_m
        self.max_m = max_m

        self.total_steps = total_steps
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        self.step_num = 0
        self.history = {}


    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out


    def get_lr(self):
        if self.step_num <= self.step_size_up:
            computed_lr = self._annealing_cos(self.initial_lr, self.max_lr, self.step_num / self.step_size_up)
        else:
            down_step_num = self.step_num - self.step_size_up
            computed_lr = self._annealing_cos(self.max_lr, self.min_lr, down_step_num / self.step_size_down)

        return computed_lr


    def get_momentum(self):
        if self.step_num <= self.step_size_up:
            computed_momentum = self._annealing_cos(self.max_m, self.base_m, self.step_num / self.step_size_up)
        else:
            down_step_num = self.step_num - self.step_size_up
            computed_momentum = self._annealing_cos(self.base_m, self.max_m, down_step_num / self.step_size_down)

        return computed_momentum


    def on_train_begin(self, logs=None):
        if self.cyclical_momentum == True:
            for m_name in ['momentum', 'beta_1']:
                if hasattr(self.model.optimizer, m_name):
                    self.m_name = m_name
                    break


    def on_train_batch_begin(self, batch, logs=None):       
        logs = logs or {}

        self.history.setdefault('iterations', []).append(self.step_num)

        lr = self.get_lr()
        K.set_value(self.model.optimizer.lr, lr)
        self.history.setdefault('lr', []).append(lr)

        if self.cyclical_momentum == True:
            momentum = self.get_momentum()
            K.set_value(getattr(self.model.optimizer, self.m_name), momentum)
            self.history.setdefault(self.m_name, []).append(momentum)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.step_num += 1
