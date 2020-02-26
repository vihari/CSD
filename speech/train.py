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
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle

import input_data
import speech_models as models
from tensorflow.python.platform import gfile
import losses

from tensorflow.python import debug as tf_debug

slim = tf.contrib.slim
FLAGS = None


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession()
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  
  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
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
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  set_size = audio_processor.set_size('training')
  tf.logging.info('Train set_size=%d', set_size)

  lr = float(FLAGS.learning_rate)
  FLAGS.how_many_training_steps = (FLAGS.how_many_epochs * set_size)//FLAGS.batch_size
  print ("Running for %d training steps with lr: %f" % (FLAGS.how_many_training_steps, lr))
  
  # training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  # learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  training_steps_list = [FLAGS.how_many_training_steps, 1200]
  learning_rates_list = [lr, 1e-4]
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  net = lambda scope: lambda inp: models.create_model(inp, model_settings, FLAGS.model_architecture, is_training=True, scope=scope)

  # Define loss and optimizer
  ground_truth_label = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')
  ground_truth_style = tf.placeholder(
      tf.float32, [None, uid_count], name='groundtruth_style')

  # the supervised bunch
  if FLAGS.model in ["cg", "ucgv3", "ucgv6", "ucgv7", "ucgv8", "ucgv8_wosx", "ucgv11", "ucgv12", "ucgv13"]:
    with tf.variable_scope('crossgrad'):
      label_net_fn = net('label')
      style_net_fn = net('style')

      label_embedding = label_net_fn(fingerprint_input)
      style_embedding = style_net_fn(fingerprint_input)
        
    scales = {"epsilon": FLAGS.epsilon, "alpha": FLAGS.alpha}

    if FLAGS.model == "cg":
      cross_entropy, label_logits = losses.cg_losses(label_embedding, style_embedding, ground_truth_style,
                                       ground_truth_label, uid_count, label_count,
                                       fingerprint_input, scales, label_net_fn=label_net_fn, style_net_fn=style_net_fn)
  elif FLAGS.model == "mos" or FLAGS.model == 'mos2' or FLAGS.model == "simple":
    with tf.variable_scope('crossgrad'):
      label_net = net('label')
      label_embedding = label_net(fingerprint_input)

    if FLAGS.model == "mos":
        cross_entropy, _, label_logits = losses.mos(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS)
    elif FLAGS.model == "mos2":
        cross_entropy, _, label_logits = losses.mos2(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS)
    elif FLAGS.model == "simple":
        cross_entropy, label_logits = losses.simple(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS)

  else:
    raise NotImplementedError('Unknown model: %s' % FLAGS.model)
  #cross_entropy = losses.dan_loss(label_embedding, ground_truth_style, uid_count, label_logits, ground_truth_label)

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Add summaries for losses.
  for loss in tf.get_collection(tf.GraphKeys.LOSSES):
    tf.summary.scalar('losses/%s' % loss.op.name, loss)

  total_loss = tf.losses.get_total_loss()
  tf.summary.scalar('total_loss', total_loss)
  
  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.MomentumOptimizer(learning_rate_input, momentum=0.9, use_nesterov=True).minimize(total_loss)
#     train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(total_loss)
#     train_step = tf.train.AdadeltaOptimizer(learning_rate_input).minimize(total_loss)
    emb_mat = [var for var in tf.trainable_variables() if var.name.find('emb_mat') >= 0]
      
  predicted_indices = tf.argmax(label_logits, 1)
  expected_indices = tf.argmax(ground_truth_label, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.contrib.framework.get_or_create_global_step()  
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  print (tf.trainable_variables())
  sys.stdout.flush()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)
  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # Training loop.  
  training_steps_max = np.sum(training_steps_list)
  sess.graph.finalize()
  best_ind_val_acc, total_accuracy = -1, -1
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth, train_uids = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy, total_loss, train_step,
            increment_global_step
        ],
        feed_dict={
          fingerprint_input: train_fingerprints,
          ground_truth_label: train_ground_truth,
          ground_truth_style: train_uids, 
          learning_rate_input: learning_rate_value
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('ind-validation')
      total_accuracy = 0
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth, validation_uids = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'ind-validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_label: validation_ground_truth,
                ground_truth_style: validation_uids, 
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

    # Save the model checkpoint periodically.
#     if (training_step % FLAGS.save_step_interval == 0 or
#         training_step == training_steps_max):
    if total_accuracy > best_ind_val_acc:
      best_ind_val_acc = total_accuracy
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth, test_uids = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_label: test_ground_truth,
            ground_truth_style: test_uids, 
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))


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
      default=10,
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
      '--how_many_epochs',
      type=int,
      default=100,
      help='How many training epochs to run',)
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=18000,
      help='How many training epochs to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
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
      '--save_step_interval',
      type=int,
      default=400,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
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
    '--lmbda',
    type=float,
    default=0.5,
    help='MoS hyperparam, scale for specialized loss')
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
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
