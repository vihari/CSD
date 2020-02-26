# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import speech_models as models
from tensorflow.python.platform import gfile
import losses
import tqdm

from tensorflow.python import debug as tf_debug

slim = tf.contrib.slim
FLAGS = None


def eval_simple(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession()

  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.training_percentage,
      FLAGS.validation_percentage, FLAGS.testing_percentage, model_settings)
  
#   val_fname, test_fname = 'extra-data/all_paths.txt', 'extra-data/all_paths.txt'
#   audio_processor = input_data.make_processor(val_fname, test_fname, FLAGS.wanted_words.split(','), FLAGS.data_dir, model_settings)
#   audio_processor.num_uids = FLAGS.training_percentage + 2
  
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  # uid_count = audio_processor.num_uids
  uid_count = audio_processor.num_uids
  
  print ("Label count: %d uid count: %d" % (label_count, uid_count));
  #sys.exit()
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  net = lambda scope: lambda inp: models.create_model(inp, model_settings, FLAGS.model_architecture, is_training=True, scope=scope)

  # Define loss and optimizer
  ground_truth_label = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')
  ground_truth_style = tf.placeholder(
      tf.float32, [None, uid_count], name='groundtruth_style')

  with tf.variable_scope('crossgrad'):
    label_net = net('label')
    label_embedding = label_net(fingerprint_input)
  
  _, label_logits = losses.simple(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS)
    
  predicted_indices = tf.argmax(label_logits, 1)
  expected_indices = tf.argmax(ground_truth_label, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  global_step = tf.contrib.framework.get_or_create_global_step()  
  increment_global_step = tf.assign(global_step, global_step + 1)

  lvars = [var for var in tf.global_variables() if var.name.find("tune_var")<0]
  saver = tf.train.Saver(lvars)
  
  tf.global_variables_initializer().run()

  print (tf.trainable_variables())
  sys.stdout.flush()

  if FLAGS.model_dir:
    saver.restore(sess, FLAGS.model_dir)
    start_step = global_step.eval(session=sess)

  sess.graph.finalize()

  set_size = audio_processor.set_size('training')
  tf.logging.info('Train set_size=%d', set_size)

  set_size = audio_processor.set_size('testing')
  tf.logging.info('Test set_size=%d', set_size)

  with open("results.org", "a") as f:
    f.write(FLAGS.model_dir + "\n")
    f.write("|simple|")

  for split in ['validation', 'testing']:
    all_acc = []
    _i = 0
    num_correct, total = 0, 0
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(-1, 0, model_settings, 0.0, 0, 0, split, sess)
    test_accuracy = sess.run(evaluation_step, feed_dict={fingerprint_input: test_fingerprints, ground_truth_label: test_ground_truth})
    print ("Accuracy: %f" % test_accuracy)
    
#     for test_fingerprints, test_ground_truth in audio_processor.get_data_per_domain(model_settings, 0.0, 0.0, 0, split, sess):
#       label_dist = np.sum(test_ground_truth, axis=0)
#       if label_dist[0] > 0:
#         continue

# #       print ("Num samples: %d label dist: %s" % (len(test_fingerprints), label_dist))
#       _ln = len(test_ground_truth)

#       test_accuracy, conf_matrix = sess.run(
#           [evaluation_step, confusion_matrix],
#           feed_dict={
#               fingerprint_input: test_fingerprints,
#               ground_truth_label: test_ground_truth,
#           })
# #       print ("Acc: %f" % test_accuracy)
#       all_acc.append(test_accuracy)
#       num_correct += (test_accuracy*len(test_ground_truth))
#       total += len(test_ground_truth)
#       _i += 1
# #     print ("Avg acc over %d domains is: %f" % (_i, np.mean(all_acc)))
#     print ("Avg acc over %d domains is: %f" % (_i, num_correct/total))
#     with open("results.org", "a") as f:
#       f.write("%d|%f|" % (_i, 100*np.mean(all_acc)))
  with open("results.org", "a") as f:
    f.write("\n")


def eval_mos(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession()

  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.training_percentage,
      FLAGS.validation_percentage, FLAGS.testing_percentage, model_settings)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  # uid_count = audio_processor.num_uids
  uid_count = audio_processor.num_uids
  
  print ("Label count: %d uid count: %d" % (label_count, uid_count));
  #sys.exit()
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  net = lambda scope: lambda inp: models.create_model(inp, model_settings, FLAGS.model_architecture, is_training=True, scope=scope)

  # Define loss and optimizer
  ground_truth_label = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')
  ground_truth_style = tf.placeholder(
      tf.float32, [None, uid_count], name='groundtruth_style')

  with tf.variable_scope('crossgrad'):
    label_net = net('label')
    label_embedding = label_net(fingerprint_input)

  with tf.variable_scope(''):
    cross_entropy, label_logits = losses.mos(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS)
  tune_var = tf.get_variable("tune_var", [FLAGS.num_uids])
  c_wts = tf.sigmoid(tune_var)
  c_wts /= tf.norm(c_wts)
#   c_wts /= tf.reduce_sum(c_wts)

#   tune_var = tf.get_variable("tune_var", [uid_count])
  with tf.variable_scope('', reuse=True):
    emb_mat = tf.get_variable("emb_mat", [uid_count, FLAGS.num_uids])
#     _cwts = tf.nn.softmax(tune_var)
#     c_wts = tf.einsum("j,ju->u", _cwts, emb_mat)
    
    logits_for_tune = losses.mos_tune(label_embedding, c_wts, ground_truth_label, label_count, fingerprint_input, FLAGS)
    # look away, ugly code!!
    logits_dir = losses.mos_tune(label_embedding, c_wts, ground_truth_label, label_count, fingerprint_input, FLAGS)
    
#     logits_for_tune = tf.nn.relu(logits_for_tune)
#     logits_dir = tf.nn.relu(logits_dir)
    
  probs_for_tune = tf.nn.softmax(logits_for_tune, axis=1)
  probs_dir = tf.nn.softmax(logits_dir, axis=1)
  agg_prob = tf.reduce_sum(probs_for_tune, axis=0)
  agg_prob2 = tf.reduce_sum(probs_dir, axis=0)
  # the first label is silence which is not present for many ids
  np_u = np.ones([label_count], dtype=np.float32)/(label_count - 1)
  np_u[0] = 0
  U = tf.constant(np_u)

  _l = logits_for_tune - tf.expand_dims(tf.reduce_min(logits_for_tune, axis=1), axis=1)
#   _l /= tf.expand_dims(tf.reduce_max(_l, 1), 1)
  loss1 = tf.reduce_sum(tf.abs(agg_prob - (U*tf.cast(tf.shape(probs_for_tune)[0], tf.float32))))
  loss2 = -tf.reduce_mean(tf.reduce_sum(probs_for_tune * tf.one_hot(tf.argmax(probs_for_tune, axis=1), depth=label_count), axis=1))
  loss4 = -tf.reduce_mean(tf.reduce_sum(probs_for_tune*(_l), axis=1))
  loss_for_tune = loss4
  
  _l = logits_dir - tf.expand_dims(tf.reduce_min(logits_dir, axis=1), axis=1)
  _l /= tf.expand_dims(tf.reduce_max(_l, 1), 1)
  loss1 = tf.reduce_sum(tf.abs(agg_prob2 - (U*tf.cast(tf.shape(probs_for_tune)[0], tf.float32))))
  loss2 = -tf.reduce_mean(tf.reduce_sum(probs_dir * tf.one_hot(tf.argmax(probs_dir, axis=1), depth=label_count), axis=1))
  loss4 = -tf.reduce_mean(tf.reduce_sum(probs_dir*(_l), axis=1))
  loss_dir = loss4

  predicted_tuned_indices = tf.argmax(logits_for_tune, axis=1)
  tune_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_tuned_indices, tf.argmax(ground_truth_label, 1)), tf.float32))
  
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]
  
  predicted_indices = tf.argmax(label_logits, 1)
  expected_indices = tf.argmax(ground_truth_label, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  global_step = tf.contrib.framework.get_or_create_global_step()  
  increment_global_step = tf.assign(global_step, global_step + 1)

  lvars = [var for var in tf.global_variables() if var.name.find("tune_var")<0]
  saver = tf.train.Saver(lvars)

  opt = tf.train.MomentumOptimizer(0.1, 0.1, use_nesterov=True)
#   opt = tf.train.AdamOptimizer(0.01)
#   opt = tf.train.GradientDescentOptimizer(0.2)
  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    tune_op = opt.minimize(loss_for_tune, var_list=[tune_var])
  
  tf.global_variables_initializer().run()

  print (tf.trainable_variables())
  sys.stdout.flush()

  if FLAGS.model_dir:
    saver.restore(sess, FLAGS.model_dir)
    start_step = global_step.eval(session=sess)

  ph_uidx = tf.placeholder(tf.int32, [])
  tf_emb = tf.nn.embedding_lookup(emb_mat, [ph_uidx])
  tf_emb = tf.sigmoid(tf_emb)
  tf_emb /= tf.expand_dims(tf.norm(tf_emb, axis=1), axis=1)
#   sess.graph.finalize()

  set_size = audio_processor.set_size('training')
  tf.logging.info('set_size=%d', set_size)

# train accuracy is almost 100.

#   total_accuracy = 0
#   total_conf_matrix = None
#   for i in xrange(0, set_size, FLAGS.batch_size):
#     test_fingerprints, test_ground_truth, test_uids = audio_processor.get_data(
#         FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'training', sess)
#     test_accuracy, conf_matrix = sess.run(
#         [evaluation_step, confusion_matrix],
#         feed_dict={
#             fingerprint_input: test_fingerprints,
#             ground_truth_label: test_ground_truth,
#             ground_truth_style: test_uids, 
#         })
#     batch_size = min(FLAGS.batch_size, set_size - i)
#     total_accuracy += (test_accuracy * batch_size) / set_size
#     if total_conf_matrix is None:
#       total_conf_matrix = conf_matrix
#     else:
#       total_conf_matrix += conf_matrix
#   tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
#   tf.logging.info('Final train accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))

  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  #   for i in xrange(0, set_size, FLAGS.batch_size):
  _t1, _t2, _t3 = 0, 0, 0
  _i1, _i2, _i3 = 0, 0, 0
  agg_acc = np.zeros([uid_count])
  for test_fingerprints, test_ground_truth in audio_processor.get_data_per_domain(model_settings, 0.0, 0.0, 0, 'validation', sess):
    sess.run(tune_var.initializer)
    label_dist = np.sum(test_ground_truth, axis=0)
    if label_dist[0] > 0 or _i1 >= 20:
      continue
        
    print ("Num samples: %d label dist: %s" % (len(test_fingerprints), label_dist))
    _ln = len(test_ground_truth)
        
    max_test = -1
    ruid = np.random.randint(0, uid_count)
    all_acc = []
    for uidx in tqdm.tqdm(range(uid_count)):
      uids = np.zeros([_ln, uid_count])
      uids[np.arange(_ln), uidx] = 1

      test_accuracy, conf_matrix = sess.run(
          [evaluation_step, confusion_matrix],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_label: test_ground_truth,
              ground_truth_style: uids, 
          })
      all_acc.append(test_accuracy)
      
      if test_accuracy > max_test:
        max_test = test_accuracy
        best_wt = sess.run(tf_emb, feed_dict={ph_uidx: uidx})[0]
      #print ("Test acc: %0.4f" % test_accuracy)
      if uidx == ruid:
        base = test_accuracy
    
    _t3 += base
    _i3 += 1
    
    _t1 += max_test
    _i1 += 1.

    agg_acc += np.array(sorted(all_acc))
    print ("Base Test acc: %0.4f" % (base))
    print ("Best Test acc: %0.4f -- wt: %s" % (max_test, best_wt))
    
    fd={fingerprint_input: test_fingerprints,
        ground_truth_label: test_ground_truth,
       }
#     sess.run(tf.assign(tune_var, best_wt))
#     np_loss_for_best = sess.run(loss_dir, feed_dict=fd)
#     print ("Loss for best wt: %f" % np_loss_for_best)
    sess.run(tune_var.initializer)
    for _it in range(100):
      _, np_l, np_wts, np_wts2 = sess.run([tune_op, loss_for_tune, c_wts, tune_var], feed_dict=fd)
      if _it % 100 == 0:
        print ("Loss: %f wts: %s %s" % (np_l, np_wts, np_wts2)) 
    np_tuned_acc, np_preds = sess.run([tune_acc, tf.one_hot(predicted_tuned_indices, label_count)], feed_dict=fd)
    
    _t2 += np_tuned_acc
    _i2 += 1
    print ("Tuned acc: %f dist: %s" % (100*np_tuned_acc, np.sum(np_preds, axis=0)))
    #print (conf_matrix)
  print ("Defau Avg test accuracy: %f over %d domains" % ((_t3/_i3), _i3))  
  print ("Brute Avg test accuracy: %f over %d domains" % ((_t1/_i1), _i1))  
  print ("Tuned Avg test accuracy: %f over %d domains" % ((_t2/_i2), _i2))
  agg_acc /= _i1
  for pi in range(0, 110, 10):
    print (pi, np.percentile(agg_acc, pi))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='./speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)

  parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help="""\
    Random seed.
    """
  )
  #default='/tmp/speech_dataset/',
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--training_percentage',
      type=int,
      default=100,
      help='What percentage of wavs to use as a training set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='18000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
    '--model',
    type=str,
    default='cg',
    help='Model to be used for training.')
  parser.add_argument(
    '--FGSM',
    action='store_true',
    default=False, 
    help='Whether or not to use FGSM perturbs')
  parser.add_argument(
    '--CG',
    action='store_true',
    default=False, 
    help='Whether or not to use CG perturbs')
  
  parser.add_argument(
    '--alpha',
    type=float,
    default=0,
    help='The mixing coefficient for perturbation loss')
  parser.add_argument(
    '--epsilon',
    type=float,
    default=0,
    help='The perturbation scalar')
  parser.add_argument(
    '--num_uids',
    type=int,
    default=-1,
    help='Number of style labels to model')
  parser.add_argument(
    '--v8_explicit_perturbs',
    action='store_true',
    default=False, 
    help='To use explicit or latent perturbs used in ucrossv8 routine in losses.py')
  parser.add_argument(
    '--v8_dfunc',
    type=str,
    default='', 
    help='One of [L1], used in v8; style perturbs are obtained by gradients on this distance function.')
  parser.add_argument(
    '--v2_no_g_recon',
    action='store_true',
    default=False, 
    help='Used by ucrossv2_wogating, whether or not to impose reconstruction loss on g')
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.set_random_seed(FLAGS.seed)
  print ("\n\n************")
  print ("Model: %s \nalpha: %f \nepsilon: %f \nnum style: %d \nFGSM: %d \nCG: %d\nv8_explicit_perturbs:%d\nv8_dfunc: %s\nv2_no_g_recon: %s\nseed: %f\n" % (FLAGS.model, FLAGS.alpha, FLAGS.epsilon, FLAGS.num_uids, FLAGS.FGSM, FLAGS.CG, FLAGS.v8_explicit_perturbs, FLAGS.v8_dfunc, FLAGS.v2_no_g_recon, FLAGS.seed))
  print ("**************\n")
  tf.app.run(main=eval_simple, argv=[sys.argv[0]] + unparsed)
