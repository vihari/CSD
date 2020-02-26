# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf
import tqdm
from scipy import misc
from scipy.ndimage import rotate as rot

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS


def prepare_data(leftout_angles):
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
  TRAIN_SIZE = 1000
  TEST_SIZE = 1000
  
  np.random.seed(0)
  idxs = np.random.choice(np.arange(len(train_images)), TRAIN_SIZE, replace=False)
  idxs2 = np.random.choice(np.arange(len(test_images)), TEST_SIZE, replace=False)
  train_images = train_images[idxs].astype(np.float32)
  train_labels = train_labels[idxs].tolist()
  test_images = test_images[idxs2].astype(np.float32)
  test_labels = test_labels[idxs2].tolist()
  
  train_images = (train_images - 128.)/128.
  test_images = (test_images - 128.)/128.
          
  # transform all train and test images
  _train_images, _train_labels, _train_uids = [], [], []
  _test_images, _test_labels, _test_uids = [], [], []
  for ai, angle in enumerate(range(0, 90, 15)):        
    if angle in leftout_angles:
      _timgs = []
      for ti in tqdm.tqdm(range(len(test_images)), desc="Transforming test images"):
        _tr = test_images[ti]
        _timgs.append(rot(_tr, angle, reshape=False))

      _test_images += _timgs
      _test_labels += test_labels
      _test_uids += [ai-1]*len(test_images)
    else:
      _timgs = []
      for ti in tqdm.tqdm(range(len(train_images)), desc="Transforming train images"):
        _tr = train_images[ti]
        _timgs.append(rot(_tr, angle, reshape=False))

      _train_images += _timgs
      _train_labels += train_labels
      _train_uids += [ai-1]*len(train_images)
    
  train_images, train_labels, train_uids = np.array(_train_images), np.array(_train_labels), np.array(_train_uids)
  test_images, test_labels, test_uids = np.array(_test_images), np.array(_test_labels), np.array(_test_uids)
  
  train = (train_images, train_labels, train_uids)
  test = (test_images, test_labels, test_uids)
  print (np.max(train[0]), np.min(train[0]))
  print (np.max(test[0]), np.min(test[0]))

  print ("Num Train: %d num test: %d" % (len(train_images), len(_test_images)))
  return train, test, test


def prepare_data2():
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

  TRAIN_SIZE = -1
  TEST_SIZE = -1
  np.random.seed(2)
  
  if TRAIN_SIZE > 0:
    idxs = np.random.choice(np.arange(len(train_images)), TRAIN_SIZE, replace=False)
    train_images = train_images[idxs]
    train_labels = train_labels[idxs].tolist()
  
  if TEST_SIZE > 0:
    idxs = np.random.choice(np.arange(len(test_images)), TEST_SIZE, replace=False)
    test_images = test_images[idxs]
    test_labels = test_labels[idxs]
  
  train = (np.array(train_images), np.array(train_labels), np.zeros(len(train_labels)))
  test = (np.array(test_images), np.array(test_labels), np.zeros(len(test_labels)))
  print (np.shape(train[0]))
  print (np.shape(test[0]))
  return train, test

def prepare_data_for(angle, DEF=0):
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  np.random.seed(0)
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

  TRAIN_SIZE = 1000
  TEST_SIZE = 1000
  
  idxs = np.random.choice(np.arange(len(train_images)), TRAIN_SIZE, replace=False)
  train_images = train_images[idxs].astype(np.float32)
  train_labels = train_labels[idxs].tolist()
  
  idxs = np.random.choice(np.arange(len(test_images)), TEST_SIZE, replace=False)
  test_images = test_images[idxs].astype(np.float32)
  test_labels = test_labels[idxs].tolist()
#   test_labels = test_labels.tolist()
  
  # transform all train and test images
  _train_images, _train_labels, _train_uids = [], [], []
  train_per_domain, test_per_domain = {}, {}
  
  _timgs, _labels = [], []
  for angle in range(15, 90, 15):
    for ti in tqdm.tqdm(range(len(train_images)), desc="Transforming train images"):
      _timgs.append(rot(train_images[ti], angle, reshape=False))
    _labels += train_labels
    
  train = [np.array(_timgs), np.array(_labels), np.array([DEF]*len(_timgs))]
  
  _timgs = []
  _labels = []
  angles = [_ for _ in range(-20, 15, 5)]
  angles += [_ for _ in range(80, 125, 5)]
  for angle in angles:
    for ti in tqdm.tqdm(range(len(test_images)), desc="Transforming test images"):
      _timgs.append(rot(test_images[ti], angle, reshape=False))
    _labels += test_labels
  test = [np.array(_timgs), np.array(_labels), np.array([DEF]*len(_labels))]

  return train, test
