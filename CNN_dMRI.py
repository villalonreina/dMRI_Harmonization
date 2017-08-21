#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:51:56 2017

@author: jvillalo
"""

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution2D, \
    MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D, merge, Reshape
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.layers import concatenate, add, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
# from keras.utils.visualize_util import plot
from keras.callbacks import Callback
from keras import backend as K
from keras.utils import to_categorical as to_cat
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import itertools
import imageio
from numpy.lib.stride_tricks import as_strided
from skimage.util import view_as_windows
import os
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from scipy.ndimage.interpolation import affine_transform
import transformations as t
import math

mpl.use('Agg')
# configures TensorFlow to not try to grab all the GPU memory
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

scratch_dir = '/data1/data/iSeg-2017/'
input_file = scratch_dir + 'baby-seg.hdf5'

category_mapping = [0, 10, 150, 250]
img_shape = (144, 192, 128)
n_tissues = 4
patch_shape = (48, 48, 48)


def lr_scheduler(model):
    # reduce learning rate by factor of 10 every 100 epochs
    def schedule(epoch):
        new_lr = K.get_value(model.optimizer.lr)

        if epoch % 200 == 0:
            new_lr = new_lr / 2

        return new_lr

    scheduler = LearningRateScheduler(schedule)
    return scheduler


def model_checkpoint(filename):
    return ModelCheckpoint(scratch_dir + 'filename', save_best_only=True, save_weights_only=True)


def train_tl():
    from neuroembedding import autoencoder, encoder, t_net, tl_net

    training_indices = list(range(0, 10))
    validation_indices = [9]
    testing_indices = list(range(10, 24))
    ibis_indices = list(range(24, 53))

    autoencoder = autoencoder()

    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)

    autoencoder.compile(optimizer=sgd, loss='categorical_crossentropy')

    hist_one = autoencoder.fit_generator(
        label_batch(training_indices),
        len(training_indices),
        epochs=200,
        verbose=1,
        callbacks=[lr_scheduler(autoencoder), model_checkpoint('autoencoder.hdf5')],
        validation_data=[label_batch(validation_indices)],
        validation_steps=[len(validation_indices)]
    )   # training autoencoder should be complete here

    enc_model = encoder()
    enc_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    enc_model.load_weights(scratch_dir + 'autoencoder.hdf5', by_name=True)

    encoded_shape = enc_model.get_output_shape_at('enc')
    print('shape of encoded label space', encoded_shape)

    f2 = h5py.File(scratch_dir + 'encoded_labels.hdf5', 'w')
    f2.create_dataset('label_encoding', (len(training_indices) +
                                         len(validation_indices) +
                                         len(ibis_indices), encoded_shape),
                      dtype='float32')

    label_encoding = f2['label_encoding']

    f = h5py.File(input_file)
    labels = f['labels']
    images = f['images']

    for i in training_indices + validation_indices + ibis_indices:
        predicted = enc_model.predict(labels[i, ...][np.newaxis, ...], batch_size=1)
        segmentation = from_categorical(predicted, category_mapping)
        label_encoding[i, ...] = segmentation

    t_net = t_net()
    t_net.load_weights(scratch_dir + 'autoencoder.hdf5', by_name=True)
    t_net.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    hist_two = t_net.fit_generator(
        batch(training_indices),
        len(training_indices),
        epochs=500,
        verbose=1,
        callbacks=[lr_scheduler, model_checkpoint('t_net.hdf5')],
        validation_data=batch(validation_indices),
        validation_steps=len(validation_indices)
    )

    tl_net = tl_net()
    tl_net.load_weights(scratch_dir + 't_net.hdf5')
    tl_net.compile(optimizer=Adam(lr=1e-8), loss=dice_coef_loss, metrics=[dice_coef])

    hist_three = tl_net.fit(
        [images[training_indices, :, :, :][np.newaxis, ...], labels[training_indices, :, :, :][np.newaxis, ...]],
        [labels[training_indices, :, :, :][np.newaxis, ...]],
        batch_size=1,
        epochs=100,
        verbose=1,
        callbacks=[lr_scheduler, model_checkpoint('tl_net.hdf5')],
        validation_data=[images[validation_indices, :, :, :][np.newaxis, ...], labels[validation_indices, :, :, :][np.newaxis, ...]]
    )


###########################
#
#   LOSS FUNCTIONS
#
###########################


def dice_coef(y_true, y_pred):
    """ DICE coefficient: 2TP / (2TP + FP + FN). An additional smoothness term is added to ensure no / 0
    :param y_true: True labels.
    :type: TensorFlow/Theano tensor.
    :param y_pred: Predictions.
    :type: TensorFlow/Theano tensor of the same shape as y_true.
    :return: Scalar DICE coefficient.
    """
    # exclude the background class from DICE calculation

    score = 0

    # category_weight = [1.35, 17.85, 8.27*10, 11.98*10]

    category_weight = [1, 1, 1, 1]

    for i, (c, w) in enumerate(zip(category_mapping, category_weight)):
        score += w*(2.0 * K.sum(y_true[..., i] * y_pred[..., i]) / (K.sum(y_true[..., i]) + K.sum(y_pred[..., i])))

    return score / np.sum(category_weight)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def final_dice_score(y_true, y_pred):
    dice = {}
    for i, c in enumerate(zip(category_mapping)):
        dice[str(c)] = (2.0 * K.sum(y_true[..., i] * y_pred[..., i]) / (K.sum(y_true[..., i]) + K.sum(y_pred[..., i])))

    return dice


def to_categorical(y):
    """Converts a class vector (integers) to binary class matrix.
    Keras function did not support sparse category labels
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    categories = sorted(set(np.array(y, dtype="uint8").flatten()))
    num_classes = len(categories)
    # print(categories)

    cat_shape = np.shape(y)[:-1] + (num_classes,)
    categorical = np.zeros(cat_shape, dtype='b')

    for i, cat in enumerate(categories):
        categorical[..., i] = np.equal(y[..., 0], np.ones(np.shape(y[..., 0]))*cat)
        # categorical[y == cat] = 1

    return categorical


def patch_label_categorical(y):
    n_samples = y.shape[0]
    cat_shape = (n_samples,) + (n_tissues,)
    categorical = np.zeros(cat_shape, dtype='b')

    for i, cat in enumerate(category_mapping):
        categorical[..., i] = np.equal(y[..., 0], np.ones(np.shape(y[..., 0]))*cat)

    return categorical


def from_categorical(categorical, category_mapping):
    """Combines several binary masks for tissue classes into a single segmentation image
    :param categorical:
    :param category_mapping:
    :return:
    """
    img_shape = np.shape(categorical)[1:-1]
    cat_img = np.argmax(np.squeeze(categorical), axis=-1)

    segmentation = np.zeros(img_shape, dtype='uint8')

    for i, cat in enumerate(category_mapping):
        segmentation[cat_img == i] = cat

    return segmentation


def from_categorical_patches(categorical, category_mapping):
    n_patches = categorical.shape[0]

    categories = np.zeros(n_patches, dtype='uint8')

    for i, cat in enumerate(category_mapping):
        pass

    return categories


def batch(indices, augmentMode=None):
    """
    :param indices: List of indices into the HDF5 dataset to draw samples from
    :return: (image, label)
    """
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    return_imgs = np.zeros(img_shape + (2,))

    while True:
        np.random.shuffle(indices)
        for i in indices:
            t1_image = np.asarray(images[i, ..., 0], dtype='float32')
            t2_image = np.asarray(images[i, ..., 1], dtype='float32')

            try:
                true_labels = labels[i, ..., 0]

                if augmentMode is not None:
                    if 'flip' in augmentMode:
                        # flip images
                        if np.random.rand() > 0.5:
                            t1_image = np.flip(t1_image, axis=0)
                            t2_image = np.flip(t2_image, axis=0)
                            true_labels = np.flip(true_labels, axis=0)

                    if 'affine' in augmentMode:

                        if np.random.rand() > 0.5:
                            scale = 1 + (np.random.rand(3) - 0.5) * 0.1  # up to 5% scale
                        else:
                            scale = None

                        if np.random.rand() > 0.5:
                            shear = (np.random.rand(3) - 0.5) * 0.2  # sheer of up to 10%
                        else:
                            shear = None

                        if np.random.rand() > 0.5:
                            angles = (np.random.rand(3) - 0.5) * 0.1 * 2*math.pi  # rotation up to 5 degrees
                        else:
                            angles = None

                        trans_mat = t.compose_matrix(scale=scale, shear=shear, angles=angles)
                        trans_mat = trans_mat[0:-1, 0:-1]

                        t1_image = affine_transform(t1_image, trans_mat, cval=10)
                        t2_image = affine_transform(t2_image, trans_mat, cval=10)
                        # ratio_img = affine_transform(ratio_img, trans_mat, cval=10)

                        true_labels = affine_transform(true_labels, trans_mat, order=0, cval=10)  # nearest neighbour for labels

                return_imgs[..., 0] = t1_image
                return_imgs[..., 1] = t2_image

                label = to_categorical(np.reshape(true_labels, true_labels.shape + (1,)))

                yield (return_imgs[np.newaxis, ...], label[np.newaxis, ...])

            except ValueError:
                print('some sort of value error occurred')
                # print(images[i, :, :, 80:-48][np.newaxis, ...].shape)
                yield (return_imgs[np.newaxis, ...])


def label_batch(indices):
    f = h5py.File(input_file)
    labels = f['labels']

    while True:
        np.shuffle(indices)

        for i in indices:
            true_labels = labels[i, ..., 0]
            label = to_categorical(np.reshape(true_labels, true_labels.shape + (1,)))

            yield (label[np.newaxis, ...], label[np.newaxis, ...])


if __name__ == "__main__":
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']
