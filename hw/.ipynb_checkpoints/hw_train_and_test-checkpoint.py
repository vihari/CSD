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
from matplotlib.pyplot import imread
import pickle

import input_data
import lipitk
import nhcd
import rmnist
import pickle
import datasets


# Basic model parameters as external flags.
FLAGS = None
DEF = 0
NETWORK, NUM_CLASSES, IMAGE_PIXELS, IMAGE_SIZE = None, None, None, None
train, in_dev, dev, test = None, None, None, None

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
    

def placeholder_inputs(batch_size=None):
  images_placeholder = tf.placeholder(
      tf.float32, shape=(None, IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(None))
  domain_placeholder = tf.placeholder(tf.int32, shape=(None))
  return images_placeholder, labels_placeholder, domain_placeholder


def fill_feed_dict(x_ys, images_pl, labels_pl, domain_pl, batch_size=None):
  xs, ys, us = x_ys
  if batch_size is None:
    batch_size = FLAGS.batch_size
  elif batch_size == -1:
    batch_size = len(xs)
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
#   print ("Batch size: ", FLAGS.batch_size)
  def next_batch(_xs, _ys, _us, bs):
    if len(_xs) < bs:
      bs = len(_xs)
    idxs = np.random.choice(len(_xs), size=bs, replace=False)
    return _xs[idxs], _ys[idxs], _us[idxs]
  
  images_feed, labels_feed, domain_feed = next_batch(xs, ys, us, FLAGS.batch_size)
  images_feed = np.reshape(images_feed, [FLAGS.batch_size, IMAGE_PIXELS])
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      domain_pl: domain_feed,
  }
  return feed_dict


def eprint(*args):
  _str = " ".join([str(arg) for arg in args])
  sys.stderr.write("%s\n" % _str)

  
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            domain_placeholder,
            data_set):
  bs = FLAGS.batch_size
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = max(1, (len(data_set[0]) // bs))
  num_examples = steps_per_epoch * bs
  
  xs, ys, us = data_set
  for step in xrange(steps_per_epoch):
    _mx = min((step+1)*bs, len(data_set[0]))
    idxs = np.arange(step*bs, _mx, dtype=np.int32)
    _xs, _ys, _us = xs[idxs], ys[idxs], us[idxs]
    _xs = np.reshape(_xs, [(_mx - (step*bs)), IMAGE_PIXELS])
    feed_dict = {images_placeholder: _xs,
                 labels_placeholder: _ys,
                 domain_placeholder: _us}
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  eprint('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return precision, true_count


def do_eval_macro(sess,
                  eval_correct,
                  images_placeholder,
                  labels_placeholder,
                  domain_placeholder,
                  data_set):  
  agg_prec, agg_corr = 0, 0
  num = 0
  for dom in np.unique(data_set[-1]):
    idxs = np.where(np.equal(data_set[-1], dom))
    data = (data_set[0][idxs], data_set[1][idxs], data_set[2][idxs])
    
    FLAGS.batch_size = min(FLAGS.batch_size, len(data[0]))
    prec, corr = do_eval(sess, eval_correct, images_placeholder, labels_placeholder, domain_placeholder, data)
    agg_prec += prec
    agg_corr += corr
    num += 1
    
  precision, true_count = agg_prec/num, agg_corr/num
  eprint('Num domains: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num, true_count, precision))
  return precision, true_count


def run_simple():
  SEED = FLAGS.seed
  np.random.seed(SEED)
  
  tf.reset_default_graph()
  with tf.Graph().as_default():
    tf.set_random_seed(SEED)
    ph_lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder, domain_placeholder = placeholder_inputs()

    final_layer = tf.layers.Dense(NUM_CLASSES, kernel_initializer=tf.random_normal_initializer(0, 0.05))
    with tf.variable_scope(''):
      reprs = lipitk.get_reprs(images_placeholder, image_size=IMAGE_SIZE, network=NETWORK, is_training=True)
      logits = final_layer(reprs)
    with tf.variable_scope('', reuse=True):
      reprs_for_eval = lipitk.get_reprs(images_placeholder, image_size=IMAGE_SIZE, network=NETWORK, is_training=False)
      logits_for_eval = final_layer(reprs_for_eval)
    
    loss = lipitk.loss(logits, labels_placeholder, num_classes=NUM_CLASSES)
    train_op = lipitk.training(loss, ph_lr)
    
    eval_correct = lipitk.evaluation(logits_for_eval, labels_placeholder)

    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    _vars = tf.all_variables()
    saver = tf.train.Saver(var_list=_vars)

    sess = tf.Session(config=config)
    sess.run(init)
  
    best_dev, best_test = -1, -1
    best_dev_abs, best_test_abs = -1, -1
    nepochs = FLAGS.nepochs
    num_steps = (nepochs*len(train[0]))//FLAGS.batch_size
    nsteps_per_epoch = len(train[0])/FLAGS.batch_size
    
    start_lr = FLAGS.learning_rate
    for step in tqdm.tqdm(xrange(num_steps)):
      start_time = time.time()

      feed_dict = fill_feed_dict(train,
                                 images_placeholder,
                                 labels_placeholder, 
                                 domain_placeholder)
      np_lr = start_lr
      feed_dict[ph_lr] = np_lr
      _, np_loss, np_logits = sess.run([train_op, loss, logits], feed_dict = feed_dict)
      
      if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
        eprint ("Loss: ", np_loss)
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        eprint('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                domain_placeholder,
                train)
        # Evaluate against the dev set.
        in_dev_prec, in_dev_corr = do_eval(sess, 
                                     eval_correct,
                                     images_placeholder,
                                     labels_placeholder,
                                     domain_placeholder,
                                     in_dev)
        dev_prec, dev_corr = do_eval(sess, 
                                     eval_correct,
                                     images_placeholder,
                                     labels_placeholder,
                                     domain_placeholder,
                                     dev)
        test_prec, test_corr = do_eval(sess,
                                       eval_correct,
                                       images_placeholder,
                                       labels_placeholder,
                                       domain_placeholder,
                                       test)
        if dev_prec >= best_dev:
          best_dev, best_dev_abs = dev_prec, dev_corr
          best_test, best_test_abs = test_prec, test_corr
        
    print ("test prec for best dev, test acc: %f, test acc: %f" % (best_dev, best_test))
    return in_dev_prec, best_dev, best_test
    
def run_training():
  SEED = FLAGS.seed

  np.random.seed(SEED)
  emb_dim = FLAGS.emb_dim
  num_domains = max([np.max(train[2]), np.max(dev[2]), np.max(test[2])]) + 1
  tf.reset_default_graph()
  with tf.Graph().as_default():
    tf.set_random_seed(SEED)
    ph_lr = tf.placeholder(tf.float32, name="learning_rate", shape=[])
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder, domain_placeholder = placeholder_inputs(FLAGS.batch_size)

    LN = tf.keras.layers.LayerNormalization(axis=1)
    with tf.variable_scope(''):
      reprs = lipitk.get_reprs(images_placeholder, image_size=IMAGE_SIZE, network=NETWORK, is_training=True)
      reprs = LN(reprs)
      logits1, logits2, reg_loss, common_var, specialized_common_wt, _e = lipitk.inference_bottleneckv2(reprs, domain_placeholder, num_domains=num_domains, emb_dim=emb_dim, num_classes=NUM_CLASSES)
      
    with tf.variable_scope('', reuse=True):
      reprs = lipitk.get_reprs(images_placeholder, image_size=IMAGE_SIZE, network=NETWORK, is_training=False)
      reprs = LN(reprs)
      _, logits_for_eval, _, _, _, _ = lipitk.inference_bottleneckv2(reprs, domain_placeholder, num_domains=num_domains, emb_dim=emb_dim, num_classes=NUM_CLASSES)
    
    loss1 = lipitk.loss(logits1, labels_placeholder, num_classes=NUM_CLASSES)
    loss2 = lipitk.loss(logits2, labels_placeholder, num_classes=NUM_CLASSES)
    loss = (FLAGS.lmbda * loss1) + loss2
    if FLAGS.lmbda > 0:
      loss = FLAGS.lmbda*loss1 + (1 - FLAGS.lmbda)*loss2
      loss += FLAGS.alpha*reg_loss
    if FLAGS.lmbda > 1:
      loss /= FLAGS.lmbda
    
    global_step = tf.contrib.framework.get_or_create_global_step()  
    increment_global_step = tf.assign(global_step, global_step + 1)
    
    train_op = lipitk.training(loss, ph_lr)
    eval_correct = lipitk.evaluation(logits_for_eval, labels_placeholder)

    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    _vars = tf.global_variables()
    saver = tf.train.Saver(var_list=_vars, max_to_keep=20)

    sess = tf.Session(config=config)
    sess.run(init)
  
    best_acc, best_test = -1, -1
    nepochs = FLAGS.nepochs
    nsteps = (nepochs*len(train[0]))//FLAGS.batch_size
    nsteps_per_epoch = len(train[0])/FLAGS.batch_size
    start_lr = FLAGS.learning_rate
    for step in tqdm.tqdm(xrange(nsteps)):
      start_time = time.time()

      feed_dict = fill_feed_dict(train,
                                 images_placeholder,
                                 labels_placeholder, 
                                 domain_placeholder)    
      np_lr = start_lr
      feed_dict[ph_lr] = np_lr
      
      _, np_loss, _ = sess.run([train_op, loss, increment_global_step], feed_dict = feed_dict)
      all_losses = sess.run([loss1, loss2, reg_loss], feed_dict=feed_dict)
      
      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == nsteps:
        print ("Loss: ", np_loss)
        print ("Losses: ", all_losses)
        print ("Common wt: ", sess.run(common_var))
        print ("Specialized common wt: ", sess.run(specialized_common_wt))
        print ("Emb matrix: ", sess.run(_e)[:5])

        # Evaluate against the training set.
        eprint('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                domain_placeholder,
                train)
        # Evaluate against the dev set.
        in_dev_acc, _ = do_eval(sess,
                             eval_correct,
                             images_placeholder,
                             labels_placeholder,
                             domain_placeholder,
                             in_dev)
        dev_acc, _ = do_eval(sess,
                             eval_correct,
                             images_placeholder,
                             labels_placeholder,
                             domain_placeholder,
                             dev)
        test_acc, _ = do_eval(sess,
                             eval_correct,
                             images_placeholder,
                             labels_placeholder,
                             domain_placeholder,
                             test)
        
        if dev_acc >= best_acc:
          best_acc = dev_acc
          best_test = test_acc
          checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
          saver.save(sess, checkpoint_file, global_step=step)
    print ("Best in-domain dev acc: %f test: %f" % (best_acc, best_test))
    return in_dev_acc, best_acc, best_test

    
def run_cg(cgpp=False):
  SEED = FLAGS.seed
  
  np.random.seed(SEED)
  emb_dim = FLAGS.emb_dim
  num_domains = np.max(train[2]) + 1
  tf.reset_default_graph()
  with tf.Graph().as_default():
    tf.set_random_seed(SEED)
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder, domain_placeholder = placeholder_inputs(
        FLAGS.batch_size)
    ph_lr = tf.placeholder(tf.float32, name="learning_rate")

    if not cgpp:
      cg_fn = lipitk.cg
    else:
      cg_fn = lipitk.cgpp
    
    with tf.variable_scope('', ):
      loss, _, debug_print = cg_fn(images_placeholder, labels_placeholder, domain_placeholder, image_size=IMAGE_SIZE, is_training=True, network=NETWORK, num_classes=NUM_CLASSES, num_domains=num_domains, FLAGS=FLAGS)
    with tf.variable_scope('', reuse=True):
      _, logits_for_eval, __ = cg_fn(images_placeholder, labels_placeholder, domain_placeholder, image_size=IMAGE_SIZE, is_training=False, network=NETWORK, num_classes=NUM_CLASSES, num_domains=num_domains, FLAGS=FLAGS)
        
    train_op = lipitk.training(loss, ph_lr)
    eval_correct = lipitk.evaluation(logits_for_eval, labels_placeholder)

    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    _vars = tf.all_variables()
    saver = tf.train.Saver(var_list=_vars, max_to_keep=20)

    sess = tf.Session(config=config)
    sess.run(init)
  
    best_acc, best_test = -1, -1
    nepochs = FLAGS.nepochs
    nsteps = (nepochs*len(train[0]))//FLAGS.batch_size
    for step in tqdm.tqdm(xrange(nsteps)):
      start_time = time.time()

      feed_dict = fill_feed_dict(train,
                                 images_placeholder,
                                 labels_placeholder, 
                                 domain_placeholder)    
      lr = FLAGS.learning_rate
        
      feed_dict[ph_lr] = lr
      _, np_loss = sess.run([train_op, loss], feed_dict = feed_dict)
      
      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == nsteps:
        print ("Loss: ", np_loss)
        
        if debug_print is not None:
          np_dp = sess.run(debug_print, feed_dict=feed_dict)
          print ("****Debug:****")
          print (np_dp)
        
        # Evaluate against the training set.
        eprint('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                domain_placeholder,
                train)
        # Evaluate against the dev set.
        in_dev_acc, _ = do_eval(sess,
                             eval_correct,
                             images_placeholder,
                             labels_placeholder,
                             domain_placeholder,
                             in_dev)
        dev_acc, _ = do_eval(sess,
                             eval_correct,
                             images_placeholder,
                             labels_placeholder,
                             domain_placeholder,
                             dev)
        test_acc, _ = do_eval(sess,
                             eval_correct,
                             images_placeholder,
                             labels_placeholder,
                             domain_placeholder,
                             test)
        
        if dev_acc >= best_acc:
          best_acc = dev_acc
          best_test = test_acc
          checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
          saver.save(sess, checkpoint_file, global_step=step)
    print ("Best in-domain ind dev acc: %f dev acc: %f test: %f" % (in_dev_acc, best_acc, best_test))
    return in_dev_acc, best_acc, best_test

  
def main(_):
  seed = 0
  
  in_dev_accs, dev_accs, test_accs = [], [], []
  for seed in range(3):
    FLAGS.seed = seed
    np.random.seed(seed)
    
    if FLAGS.simple:
#       in_dev_acc, dev_acc, test_acc = run_simple()
      FLAGS.lmbda = 0
      in_dev_acc, dev_acc, test_acc = run_training()
    elif FLAGS.cg:
      in_dev_acc, dev_acc, test_acc = run_cg()
    else:
      in_dev_acc, dev_acc, test_acc = run_training()
    in_dev_accs.append(in_dev_acc)
    dev_accs.append(dev_acc)
    test_accs.append(test_acc)
  print ( "InD Val, Val, test acc: %0.4f (%0.4f), %0.4f (%0.4f), %0.4f (%0.4f)" % (np.mean(in_dev_accs), np.std(in_dev_accs), np.mean(dev_accs), np.std(dev_accs), np.mean(test_accs), np.std(test_accs)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      type=str,
      default='lipitk',
      help='Dataset to evaluate on.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=1e-3,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=0,
      help='Random seed.'
  )
  parser.add_argument(
      '--num_train',
      type=int,
      default=-1,
      help='Number of training domains'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=15000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--nepochs',
      type=int,
      default=100,
      help='Number of epochs'
  )
  parser.add_argument(
    '--init_ckpt',
    type=str,
    default=None,
    help='Path to initializing checkpoint'
  )
  parser.add_argument(
    '--emb_dim',
    type=int,
    default=2,
    help='Number of basis vectors'
  )
  parser.add_argument(
    '--lmbda',
    type=float,
    default=.5,
    help='Loss specific component weight'
  )
  parser.add_argument(
    '--alpha',
    type=float,
    default=1,
    help='Coeff for reg. loss'
  )
  parser.add_argument(
    '--simple',
    action='store_true',
    help='Trains and evaluates a simple baseline with no experts.'
  )
  parser.add_argument(
    '--cg',
    action='store_true',
    help='Evaluate the Crossgrad baseline.'
  )
  
  parser.add_argument(
    '--cg_eps',
    type=float,
    default=10,
    help='Step size for perturbations in CG/CG++'
  )
  
  parser.set_defaults(simple=False)
  
  FLAGS, unparsed = parser.parse_known_args()
  if not FLAGS.simple:
    FLAGS.log_dir = "lipitk_log/lipitktuner_nt=%d_fonts_e=%d_seed_%d" % (FLAGS.num_train, FLAGS.emb_dim, FLAGS.seed)
  else:
    FLAGS.log_dir = "lipitk_log/lipitktuner_wnorm_simple_nt=%d_fonts_e=%d_seed_%d" % (FLAGS.num_train, FLAGS.emb_dim, FLAGS.seed)
  os.system("mkdir %s" % FLAGS.log_dir)
  
  if FLAGS.dataset == 'lipitk':
    train, in_dev, dev, test = lipitk.prepare_data(FLAGS.num_train)
    NETWORK = 'lenet'
    NUM_CLASSES = 111
    IMAGE_PIXELS = 1024
    IMAGE_SIZE = 32
  elif FLAGS.dataset == 'nhcd':
    train, in_dev, dev, test = nhcd.prepare_data()
    NETWORK = 'lenet'
    NUM_CLASSES = nhcd.NUM_CLASSES
    IMAGE_PIXELS = nhcd.IMAGE_PIXELS
    IMAGE_SIZE = nhcd.IMAGE_SIZE
  elif FLAGS.dataset == 'english-hnd':
    train, in_dev, dev, test = datasets.load_english_hnd()
    NETWORK = 'lenet'
    NUM_CLASSES = 59
    IMAGE_PIXELS = 1024
    IMAGE_SIZE = 32
  elif FLAGS.dataset == 'english-fnt':
    train, in_dev, dev, test = datasets.load_english_fnt()
    NETWORK = 'lenet'
    NUM_CLASSES = 62
    IMAGE_PIXELS = 1024
    IMAGE_SIZE = 32
  elif FLAGS.dataset == 'rmnist':
    NETWORK = 'lenet'
    NUM_CLASSES = rmnist.NUM_CLASSES
    IMAGE_PIXELS = rmnist.IMAGE_PIXELS
    IMAGE_SIZE = rmnist.IMAGE_SIZE
    train, dev, test = rmnist.prepare_data([0])
    in_dev = dev
  
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
