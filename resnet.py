# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(filters, kernel_size=3, strides=1, act=True, **kwargs):
    def block(x):
        x = layers.Conv2D(filters, kernel_size, strides=strides,
                          padding='same', use_bias=False, kernel_initializer='he_normal', **kwargs)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        if act: x = layers.Activation('relu')(x)
        return x
    
    return block


def basic_residual(res_filters, strides=1, **kwargs):
    def block(x):
        x = conv_block(res_filters, strides=strides, **kwargs)(x)
        x = conv_block(res_filters, act=False, **kwargs)(x)
        return x
    
    return block


class AddRelu(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # saving of model does not work without specifying name:
        self.gamma = self.add_weight(name='gamma', shape=(), initializer='zeros', trainable=True)

    def call(self, inputs):
        out = inputs[0] + self.gamma * inputs[1]
        out = layers.Activation('relu')(out)
        return out


def residual_block(res_filters, strides=1, **kwargs):
    def block(x):
        shortcut = x if strides == 1 else layers.AveragePooling2D(strides)(x)
        residual = basic_residual(res_filters, strides, **kwargs)(x)
        
        shortcut_filters = shortcut.shape[-1]
        residual_filters = residual.shape[-1]
        if shortcut_filters != residual_filters:
            shortcut = conv_block(residual_filters, kernel_size=1, act=False, **kwargs)(shortcut)
        
        y = AddRelu()([shortcut, residual])
        return y
    
    return block


def residual_body(res_filters, repetitions, strides, **kwargs):
    def block(x):
        filters = res_filters
        for rep, stride in zip(repetitions, strides):
            for _ in range(rep):
                x = residual_block(filters, stride, **kwargs)(x)
                stride = 1
            filters *= 2
        
        return x
    
    return block


def stem(filter_list, strides=1, **kwargs):
    def block(x):
        stride = strides
        for filters in filter_list:
            x = conv_block(filters, strides=stride, **kwargs)(x)
            stride = 1
        return x
    
    return block


def head(classes, **kwargs):
    def block(x):
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, **kwargs)(x)
        x = layers.Activation('softmax')(x)
        return x
    
    return block