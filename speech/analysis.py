import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf

import input_data
import speech_models as models
from tensorflow.python.platform import gfile
import losses
import tqdm

from tensorflow.python import debug as tf_debug
# from sklearn.linear_model import LogisticRegression

FLAGS = None

def fine_tune(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  
  FLAGS.training_percentage = 50
  model_dir1000 = "ckpts/speech_commands_simple_1000_0.007_0/conv.ckpt-9736"
  model_dir200 = "ckpts/speech_commands_simple_200_0.005_0/conv.ckpt-8800"
  model_dir100 = "ckpts/speech_commands_simple_100_0.005_0/conv.ckpt-5200"
  model_dir50 = "ckpts/speech_commands_simple_50_0.005_0/conv.ckpt-3600"
#   model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_5_0/conv.ckpt-3200"
  model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_2_0/conv.ckpt-3200"
  model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_5_0/conv.ckpt-5880"

  sess = tf.InteractiveSession()
  FLAGS.model_dir = model_dir_mos50
  
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
  uid_count = audio_processor.num_uids
  
  print ("Label count: %d uid count: %d" % (label_count, uid_count))
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
  
  label_embedding = tf.layers.dense(label_embedding, units=128, activation=tf.nn.leaky_relu)
  label_embedding = tf.keras.layers.LayerNormalization(axis=1)(label_embedding)
  
  with tf.variable_scope('tunable'):
    label_logits = tf.layers.dense(label_embedding, units=label_count, activation=None, use_bias=False)
  final_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=ground_truth_label))
  
  predicted_indices = tf.argmax(label_logits, 1)
  expected_indices = tf.argmax(ground_truth_label, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  global_step = tf.contrib.framework.get_or_create_global_step()  
  increment_global_step = tf.assign(global_step, global_step + 1)

  lvars = [var for var in tf.global_variables() if var.name.find("tunable")<0]
  saver = tf.train.Saver(lvars)

  sm_w = [var for var in tf.global_variables() if var.name.find("tunable") > -1]
  tvars = sm_w
  opt = tf.train.GradientDescentOptimizer(0.1)
  train_step = opt.minimize(final_loss, var_list = tvars)
  
  tf.global_variables_initializer().run()

  print (tf.trainable_variables())
  sys.stdout.flush()

  if FLAGS.model_dir:
    saver.restore(sess, FLAGS.model_dir)
    start_step = global_step.eval(session=sess)

  # retain the initialization for train vars
  for tvar in tvars:
    sess.run(tvar.initializer)
    
  sess.graph.finalize()

  train_set_size = audio_processor.set_size('training')
  tf.logging.info('Train set_size=%d', train_set_size)

  set_size = audio_processor.set_size('testing')
  tf.logging.info('Test set_size=%d', set_size)

  nepochs = 50
  num_train_steps = (train_set_size * nepochs) // FLAGS.batch_size
  
  for step in range(num_train_steps):
    train_fingerprints, train_ground_truth, train_uids = audio_processor.get_data(
      FLAGS.batch_size, 0, model_settings, 0, 0, 0, 'training', sess)
    fd = {
      fingerprint_input: train_fingerprints,
      ground_truth_label: train_ground_truth,
    }
    _, np_loss = sess.run([train_step, final_loss], feed_dict=fd)
    if step % 10 == 0:
      print ("Step: %d/%d Loss: %f" % (step, num_train_steps, np_loss))
          
  for split in ['validation', 'testing']:
    all_acc = []
    _i = 0
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(-1, 0, model_settings, 0.0, 0.0, 0, split, sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_label: test_ground_truth,
        })
    print ("Avg acc over %d domains is: %f" % (_i, test_accuracy))


def check_domain_ftvs(_):
  """
  Check numerical rank of domain specific ftvs
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession()

  FLAGS.training_percentage = 10
  model_dir1000 = "/tmp/speech_commands_simple_1000_0.007,0.001_0_0_15_0/conv.ckpt-24000"
  model_dir200 = "/tmp/speech_commands_simple_200_0.01,0.001_0_0_5_0/conv.ckpt-24000"
  model_dir100 = "/tmp/speech_commands_simple_100_0.01,0.001_0_0_10_0/conv.ckpt-23200"
  
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
  
  final_loss, label_logits = losses.simple(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS)
    
  predicted_indices = tf.argmax(label_logits, 1)
  expected_indices = tf.argmax(ground_truth_label, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  global_step = tf.contrib.framework.get_or_create_global_step()  
  increment_global_step = tf.assign(global_step, global_step + 1)

  lvars = [var for var in tf.global_variables() if var.name.find("tune_var")<0]
  saver = tf.train.Saver(lvars)

  sm_w = [var for var in tf.global_variables() if var.name.find("sm_w") > -1][0]
  sm_bias = [var for var in tf.global_variables() if var.name.find("sm_bias") > -1][0]
  tvars = [sm_w, sm_bias]
  opt = tf.train.GradientDescentOptimizer(0.1)
  train_step = opt.minimize(final_loss, var_list = tvars)
  
  tf.global_variables_initializer().run()

  print (tf.trainable_variables())
  sys.stdout.flush()

  for mi, model_dir in enumerate([model_dir100, model_dir200, model_dir1000]):
    ftvs = []
    saver.restore(sess, model_dir)

    sess.graph.finalize()

    np_w = sess.run(sm_w)
    for train_fingerprints, train_ground_truth in audio_processor.get_data_per_domain(model_settings, 0.0, 0.0, 0, 'training', sess):
      train_set_size = audio_processor.set_size('training')
      # tf.logging.info('Train set_size=%d', train_set_size)

      set_size = audio_processor.set_size('testing')
      # tf.logging.info('Test set_size=%d', set_size)

      num_train_steps = 100

      for step in range(num_train_steps):
        fd = {
          fingerprint_input: train_fingerprints,
          ground_truth_label: train_ground_truth,
        }
        _, np_loss = sess.run([train_step, final_loss], feed_dict=fd)
        if step % 10 == 0:
          print ("Step: %d/%d Loss: %f" % (step, num_train_steps, np_loss))

      ftvs.append(sess.run(sm_w))
    _, s, _ = np.linalg.svd(np.array(ftvs)[:, : , 0] - np_w[:, 0])
    diffs = np.linalg.norm(np.array(ftvs) - np_w, axis=1)/np.linalg.norm(np_w)
    print ("Model dir: %s diffs %f (%f)" % (model_dir, np.mean(diffs), np.std(diffs)))
    print ("Singular values: %s" % (s/s[0]))


def check_ftvs(_):
  """
  Check ftvs for random splits of the train data
  """
  tf.logging.set_verbosity(tf.logging.INFO)

  model_dir1000 = "ckpts/speech_commands_simple_1000_0.007_0/conv.ckpt-9736"
  model_dir200 = "ckpts/speech_commands_simple_200_0.005_0/conv.ckpt-8800"
  model_dir100 = "ckpts/speech_commands_simple_100_0.005_0/conv.ckpt-5200"
  model_dir50 = "ckpts/speech_commands_simple_50_0.005_0/conv.ckpt-3600"
#   model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_5_0/conv.ckpt-3200"
#   model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_2_0/conv.ckpt-3200"
#   model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_10_0/conv.ckpt-3200"
  model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_20_0/conv.ckpt-2800"
#   model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_5_0/conv.ckpt-5880"
#   model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_2_0/conv.ckpt-4800"
#   model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_10_0/conv.ckpt-5200"
#   model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_20_0/conv.ckpt-5600"
  model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_20_1/conv.ckpt-4800"
  emb_size = 128

  for mi, (model_type, model_dir) in enumerate([(1, model_dir_mos100)]):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    FLAGS.training_percentage = 100

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

    label_embedding = tf.layers.dense(label_embedding, units=128, activation=tf.nn.leaky_relu)
    label_embedding = tf.keras.layers.LayerNormalization(axis=1)(label_embedding)
    num_latent_styles = 20
    
    if model_type == 0:
      simple_sm_w = tf.get_variable("sm_w", shape=[emb_size, label_count])
    else:
      common_var = tf.get_variable("common_var", shape=[num_latent_styles], initializer=tf.zeros_initializer)
      common_cwt = tf.nn.sigmoid(common_var)
      common_cwt /= tf.norm(common_cwt)
      mos_sm_w = tf.get_variable("sm_w", shape=[num_latent_styles, emb_size, label_count])
      common_sm = tf.einsum("j,jkl->kl", common_cwt, mos_sm_w)

    sm_w = tf.get_variable("sm_w1", shape=[emb_size, label_count])
    label_logits = tf.einsum("ik,kl->il", label_embedding, sm_w)
    final_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_label, logits=label_logits))
      
    global_step = tf.contrib.framework.get_or_create_global_step()  
    increment_global_step = tf.assign(global_step, global_step + 1)

    lvars = [var for var in tf.global_variables() if var.name.find("sm_w1")<0]
    saver = tf.train.Saver(lvars)

    tvars = [sm_w]
    opt = tf.train.GradientDescentOptimizer(0.1)
    train_step = opt.minimize(final_loss, var_list = tvars)

    tf.global_variables_initializer().run()

    print (tf.trainable_variables())
    sys.stdout.flush()

    ftvs = []
    saver.restore(sess, model_dir)
    if model_type == 0:
      np_sm_w = sess.run(simple_sm_w)
    else:
      np_sm_w = sess.run(common_sm)
    sess.run(tf.assign(sm_w, np_sm_w))

    sess.graph.finalize()

    np_w = sess.run(sm_w)
    all_x, all_y = [], []
    for train_fingerprints, train_ground_truth in audio_processor.get_data_per_domain(model_settings, 0.0, 0.0, 0, 'training', sess):
#       all_x.append(train_fingerprints)
#       all_y.append(train_ground_truth)

#     all_x, all_y = np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
#     idxs = np.arange(len(all_x))
#     np.random.shuffle(idxs)
#     all_x, all_y = all_x[idxs], all_y[idxs]

#     q = len(all_x)//10
#     for split in range(10):
#       train_fingerprints = all_x[split*q: (split+1)*q]
#       train_ground_truth = all_y[split*q: (split+1)*q]

      num_train_steps = 300

      for step in range(num_train_steps):
        fd = {
          fingerprint_input: train_fingerprints,
          ground_truth_label: train_ground_truth,
        }
        _, np_loss = sess.run([train_step, final_loss], feed_dict=fd)
        if step % 100 == 0:
          print ("Step: %d/%d Loss: %f" % (step, num_train_steps, np_loss))

      ftvs.append(sess.run(sm_w))

    _, s, _ = np.linalg.svd(np.array(ftvs)[:, : , 0] - np_w[:, 0])
    diffs = np.linalg.norm(np.array(ftvs) - np_w, axis=1)/np.linalg.norm(np_w)
    print ("Model dir: %s diffs %f (%f)" % (model_dir, np.mean(diffs), np.std(diffs)))
    print ("Singular values: %s" % (s/s[0]))
    
  
def check_feature_utility(_):
  """
  Check ftvs for random splits of the train data
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession()

  FLAGS.training_percentage = 10
  model_dir1000 = "ckpts/speech_commands_simple_1000_0.007_0/conv.ckpt-9736"
  model_dir200 = "ckpts/speech_commands_simple_200_0.005_0/conv.ckpt-8800"
  model_dir100 = "ckpts/speech_commands_simple_100_0.005_0/conv.ckpt-5200"
  model_dir50 = "ckpts/speech_commands_simple_50_0.005_0/conv.ckpt-3600"
  model_dir_mos50 = "ckpts/speech_commands_mos-random_50_0.005_0_0_5_0/conv.ckpt-3200"
  model_dir_mos100 = "ckpts/speech_commands_mos-random_100_0.005_0_0_5_0/conv.ckpt-5880"
  
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
  
  net = label_embedding
  net = tf.layers.dense(net, units=128, activation=tf.nn.leaky_relu)
  net = tf.keras.layers.LayerNormalization(axis=1)(net)
  label_embedding = net
    
  lvars = tf.global_variables()
  saver = tf.train.Saver(lvars)

  tf.global_variables_initializer().run()

  sys.stdout.flush()

  for mi, model_dir in enumerate([model_dir50, model_dir_mos50]):
    ftvs = []
    saver.restore(sess, model_dir)

    sess.graph.finalize()

    corrs = []
    for train_fingerprints, train_ground_truth in audio_processor.get_data_per_domain(model_settings, 0.0, 0.0, 0, 'training', sess):
#       all_x.append(train_fingerprints)
#       all_y.append(train_ground_truth)
    
#     all_x, all_y = np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
#     idxs = np.arange(len(all_x))
#     np.random.shuffle(idxs)
#     all_x, all_y = all_x[idxs], all_y[idxs]
    
#     q = len(all_x)//10
#     for split in range(10):
#       train_fingerprints = all_x[split*q: (split+1)*q]
#       train_ground_truth = all_y[split*q: (split+1)*q]
      
      fd = {
        fingerprint_input: train_fingerprints,
        ground_truth_label: train_ground_truth,
      }
      np_les = sess.run(label_embedding, feed_dict=fd)
      np_les = np.transpose(np_les)
      y = np.argmax(train_ground_truth, axis=1)
      _corr = [np.corrcoef(v, y)[0, 1] for v in np_les]
      corrs.append(_corr)
    print (np.shape(corrs))
    print (np.nanmean(np.abs(corrs)), np.mean(np.nanstd(np.abs(corrs), axis=0)))

    
def check_de(_):
  """
  Check if there is domain erasure as we move from representations from 100 domains to 1000 domains
  """
  # Start a new TensorFlow session.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession()

  FLAGS.training_percentage = 10
  model_dir1000 = "/tmp/speech_commands_simple_1000_0.007,0.001_0_0_15_0/conv.ckpt-24000"
  model_dir100 = "/tmp/speech_commands_simple_100_0.01,0.001_0_0_10_0/conv.ckpt-23200"
  
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

  final_loss, label_logits, label_embedding = losses.simple(label_embedding, ground_truth_style, ground_truth_label, label_count, uid_count, fingerprint_input, FLAGS, debug=True)
    
  lvars = [var for var in tf.global_variables() if var.name.find("tune_var")<0]
  saver = tf.train.Saver(lvars)
  
  tf.global_variables_initializer().run()

  sys.stdout.flush()

  for mi, model_dir in enumerate([model_dir100, model_dir1000]):
    all_embs, all_labels = [], []
    saver.restore(sess, model_dir)
    
    di = 0
    for train_fingerprints, train_ground_truth in audio_processor.get_data_per_domain(model_settings, 0.0, 0.0, 0, 'training', sess):
      np_embs = sess.run(label_logits,
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_label: train_ground_truth,
            })
      all_embs.append(np_embs)
      all_labels.append(np.ones(len(np_embs))*di)
      di += 1
    if mi == 0:
      embs100 = np.concatenate(all_embs, axis=0)
      labels100 = np.concatenate(all_labels, axis=0)
    else:
      embs1000 = np.concatenate(all_embs, axis=0)
      labels1000 = np.concatenate(all_labels, axis=0)

  print (np.shape(np_embs), np.shape(embs100))
  print ("Num domains: %d" % len(all_embs))
  idxs = np.arange(len(embs100))
  np.random.shuffle(idxs)
  embs100, labels100 = embs100[idxs, :], labels100[idxs]
  embs1000, labels1000 = embs1000[idxs, :], labels1000[idxs]
  
  print (embs100, labels100)
  clf100 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(embs100, labels100)
  print ("Clf 100: %f" % clf100.score(embs100, labels100))

  clf1000 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(embs1000, labels1000)
  print ("Clf 1000: %f" % clf100.score(embs1000, labels1000))


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
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
 
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
    '--model',
    type=str,
    default='cg',
    help='Model to be used for training.')
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.set_random_seed(FLAGS.seed)
  tf.app.run(main=check_ftvs, argv=[sys.argv[0]] + unparsed)
