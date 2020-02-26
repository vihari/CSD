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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import tqdm

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np 
import tensorflow as tf
from scipy.ndimage import rotate as rot

import input_data
import mnist

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(
      tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(x_ys, images_pl, labels_pl, styles_pl=None):
  xs, ys, ss = x_ys
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  def next_batch(_xs, _ys, _ss, bs):
    if len(_xs) < bs:
      bs = len(_xs)
    idxs = np.random.choice(len(_xs), size=bs, replace=False)
    return _xs[idxs], _ys[idxs], _ss[idxs]
  
  images_feed, labels_feed, styles_feed = next_batch(xs, ys, ss, FLAGS.batch_size)
  images_feed = np.reshape(images_feed, [FLAGS.batch_size, mnist.IMAGE_PIXELS])
  feed_dict = {
    images_pl: images_feed,
    labels_pl: labels_feed,
  }
  if styles_pl is not None:
    feed_dict[styles_pl] = styles_feed  
  
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = max(1, (len(data_set[0]) // FLAGS.batch_size))
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def prepare_rmnist_data(test_angle):
  # Get the sets of images and labels for training, test on MNIST.
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
  
  np.random.seed(1)
  idxs = np.random.choice(np.arange(len(train_images)), 1000)
  train_images = train_images[idxs]
  train_labels = train_labels[idxs]
  
  # transform all train and test images
  all_train_images, all_train_labels, all_train_styles = [], [], []
  for angle in range(6):
    angle *= 15
    if angle == test_angle:
      continue
    for ti in tqdm.tqdm(range(len(train_images)), desc="Transforming train images"):
      all_train_images.append(rot(train_images[ti], angle, reshape=False))
      all_train_labels.append(train_labels[ti])
      all_train_styles.append(angle//15)

  test_images, test_labels, test_styles = [], [], []
  for ti in tqdm.tqdm(range(len(train_images)), desc="Transforming test images"):
    test_images.append(rot(train_images[ti], test_angle, reshape=False))
    test_labels.append(train_labels[ti])
    test_styles.append(test_angle//15)
  
  np.random.seed(1)
  pidxs = np.random.permutation(np.arange(len(all_train_images)))
  all_train_images = np.array(all_train_images)[pidxs]
  all_train_labels = np.array(all_train_labels)[pidxs]
  all_train_styles = np.array(all_train_styles)[pidxs]
  
  # uncomment the following line to randomize the domain label
  np.random.shuffle(all_train_styles)
  
  train = (all_train_images, all_train_labels, all_train_styles)
  test = (np.array(test_images), np.array(test_labels), np.array(test_styles))
  
  return train, test
  
def run_training():
  NUM_STYLES = 5
  
  train, test = prepare_rmnist_data(FLAGS.test_angle)
  
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    styles_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))

    logits, net = mnist.inference(images_placeholder, mnist.NUM_CLASSES, scope="label_net")
    loss = mnist.loss(logits, labels_placeholder)
    
    style_logits, style_net = mnist.inference(images_placeholder, NUM_STYLES, scope="style_net")
    style_loss = mnist.loss(style_logits, styles_placeholder)
    
    loss += style_loss
    train_op = mnist.training(loss, FLAGS.learning_rate, FLAGS.init_ckpt is not None)

    eval_correct = mnist.evaluation(logits, labels_placeholder)
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)

    if FLAGS.init_ckpt is not None:
      saver.restore(sess, FLAGS.init_ckpt)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      feed_dict = fill_feed_dict(train,
                                 images_placeholder,
                                 labels_placeholder, 
                                 styles_placeholder)

      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                train)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                test)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MkDir(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--test_angle',
      type=float,
      default=0,
      help='Testing angle (in degrees)'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=10000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
    '--init_ckpt',
    type=str,
    default=None,
    help='Path to initializing checkpoint'
  )
  

  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.log_dir = "mnist_log/rmnist_%d" % FLAGS.test_angle
  os.system("mkdir %s" % FLAGS.log_dir)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
