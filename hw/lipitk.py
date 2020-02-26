import numpy as np
import os
import sys

from scipy import misc
import tqdm
import pickle

import tensorflow as tf
# The MNIST images are always 28x28 pixels.

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

DATA_FLDR = "data/hpl-devnagari-iso-char-offline"

"""
Helper to load lipitk data and network definition to process it.
"""

def eprint(*args):
  _str = " ".join([str(arg) for arg in args])
  sys.stderr.write("%s\n" % _str)

def load_data(fldr):
  IMAGE_SIZE = 32
  images, labels, uids = [], [], []
  
  width, height = IMAGE_SIZE, IMAGE_SIZE
  MAX_NUM_DOMAINS = 110
  uid = 0
  cache_fname = 'data/lipitk.pkl'
  if os.path.exists(cache_fname):
    images, labels, uids = pickle.load(open(cache_fname, "rb"))
  else:
    for dom in tqdm.tqdm(range(MAX_NUM_DOMAINS)):
      dom_fldr = "%s/usr_%d" % (fldr, dom)
      if not os.path.exists(dom_fldr):
        continue
      for fname in os.listdir(dom_fldr):
        if fname.find('.tiff') < 0:
          continue
        li = int(fname.split('t')[0])

        img = misc.imread(dom_fldr + "/" + fname)
        img = misc.imresize(img, (height, width))
        img = img.astype(np.float32)
        img = misc.bytescale(img)
        img = img.astype(np.uint8)   
        
        assert np.max(img) <= 255 and np.min(img) >= 0, "Max and min of image: %f %f" % (np.max(img), np.min(img))
        img = (img-128.)/128.
        assert np.max(img) != np.min(img)
        images.append(img)
        labels.append(li)
        uids.append(uid)
      uid += 1
    pickle.dump((images, labels, uids), open(cache_fname, "wb"))
  
  print ("Labels: %s uids: %s" % (labels[:10], uids[:10]))
  print ("Labels: %s uids: %s" % (labels[-10:], uids[-10:]))
  print ("Test images: ", np.max(images[0]), np.min(images[0]))
  
  print ("Read %d examples" % len(images))
  uids = np.array(uids)
  np.random.shuffle(uids)
  return np.array(images), np.array(labels), uids


def prepare_data(num_train):
  np.random.seed(1)

  imgs, labels, uids = load_data(DATA_FLDR)
  
  domain_ids = np.sort(np.unique(uids))
  
  train_domains = domain_ids[:-30]
  dev_domains = domain_ids[-30:-20]
  test_domains = domain_ids[-20:]

  if num_train > 0:
    train_domains = sorted(train_domains)
    num_train = min(num_train, len(train_domains))
    train_domains = train_domains[:num_train]
  
  train_data, dev_data, test_data = [], [], []
  uuids = np.unique(uids)
  uid_to_idx = {domain_ids[_]:_ for _ in range(len(domain_ids))}
  
  split_doms = [train_domains, dev_domains, test_domains]
  split_dats = [[], [], []]
  idxs_per_dom = {}
  for uidx, uid in enumerate(uids):
    arr = idxs_per_dom.get(uid, [])
    arr.append(uidx)
    idxs_per_dom[uid] = arr

  for si in range(len(split_doms)):
    for dom in split_doms[si]:
      idxs = idxs_per_dom[dom]
      this_imgs, this_labels = imgs[idxs].tolist(), labels[idxs].tolist()
      this_doms = [uid_to_idx[dom]] * len(this_labels)
      if len(split_dats[si]) == 0:
        split_dats[si] = [this_imgs, this_labels, this_doms]
      else:
        split_dats[si][0] += this_imgs
        split_dats[si][1] += this_labels
        split_dats[si][2] += this_doms
  
  for si in range(len(split_doms)):
    for i in range(3):
      split_dats[si][i] = np.array(split_dats[si][i])

  train, dev, test = split_dats
  train_sp = int(0.8*len(train[0]))
  
  idxs = np.arange(len(train[0]))
  np.random.shuffle(idxs)
  train = [train[0][idxs], train[1][idxs], train[2][idxs]]
  if num_train < 1000:
    in_dev = dev
  else:
    in_dev = (train[0][train_sp:], train[1][train_sp:], train[2][train_sp:])
    train = (train[0][:train_sp], train[1][:train_sp], train[2][:train_sp])
  print (np.shape(train[0]), np.shape(dev[0]), np.shape(test[0]))
  split_dats = [train, in_dev, dev, test]
  
  eprint ("Number of train, dev and test domains: %d %d %d" % (len(train_domains), len(dev_domains), len(test_domains)))
  eprint ("Number of examples per split: %d %d %d %d" % (len(split_dats[0][0]), len(split_dats[1][0]), len(split_dats[2][0]), len(split_dats[3][0])))
  eprint ("Train labels: ", np.unique(split_dats[0][1]))
  return split_dats

    
def motian(images, dropout_prob=0.5):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 32, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)

    EMB_SIZE = 84
    net = tf.layers.dense(net, 120, name='fc3', activation=tf.nn.relu)
    net = tf.layers.dropout(net, dropout_prob)
    net = tf.layers.dense(net, 84, name='fc4', activation=None)
    return net
  
def bigger_motian(images, dropout_prob=0.1):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.conv2d(net, 128, [5, 5], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    net = slim.flatten(net)

    EMB_SIZE = 128
    net = tf.layers.dense(net, 256, name='fc3', activation=tf.nn.relu)
    net = tf.layers.dropout(net, dropout_prob)
    net = tf.layers.dense(net, EMB_SIZE, name='fc4', activation=tf.nn.relu)
    return net
  
def lenet(images, dropout_prob=0.1):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1', biases_initializer=tf.glorot_uniform_initializer, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32))
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2', biases_initializer=tf.glorot_uniform_initializer, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32))
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    print (tf.shape(net))
    print (net.get_shape())
    
    EMB_SIZE = 1024
    net = tf.layers.dropout(net, dropout_prob)
    net = tf.layers.dense(net, EMB_SIZE, name='fc4', activation=tf.nn.relu)
    return net

def get_reprs(images, image_size, scope="Lenet", network="motian", is_training=False):
  batch_size = tf.shape(images)[0]
  images = tf.reshape(images, [-1, image_size, image_size, 1])
  with tf.variable_scope(scope, 'LeNet', [images]):
    if network == 'motian':
      EMB_SIZE = 84
      net = motian(images, 0.5 if is_training else 0)
    elif network == 'bigger_motian':
      EMB_SIZE = 128
      net = bigger_motian(images, 0.1 if is_training else 0)
    elif network == 'lenet':
      EMB_SIZE = 1024
      net = lenet(images, 0.1 if is_training else 0)
    else:
      raise NotImplementedError()
      
    net = tf.reshape(net, [batch_size, EMB_SIZE])
#     net = LN(net)
    return net
    
def inference_bottleneck(net, domain_placeholder, num_classes, num_domains=5, scope="LeNet", emb_dim=2):  
  batch_size = tf.shape(net)[0]
  EMB_SIZE = net.get_shape()[-1]
  with tf.variable_scope(scope, 'LeNet'):
    DIM = emb_dim
    emb_matrix = tf.get_variable("emb_matrix", shape=[num_domains, DIM], initializer=tf.random_normal_initializer)
    common_var = tf.get_variable("common_cwt", shape=[DIM], initializer=tf.zeros_initializer)
    
    common_cwt = tf.nn.sigmoid(common_var)
    common_cwt /= tf.norm(common_cwt)
    emb_matrix = tf.nn.sigmoid(emb_matrix)
    emb_matrix /= tf.expand_dims(tf.norm(emb_matrix, axis=1), 1)
    
    # Batch size x DIM
    c_wts = tf.nn.embedding_lookup(emb_matrix, domain_placeholder)
    c_wts = tf.reshape(c_wts, [batch_size, DIM])
    
    sms = tf.get_variable("sm_matrices", shape=[DIM, EMB_SIZE, num_classes], trainable=True)
    sm_biases = tf.get_variable("sm_bias", shape=[DIM, num_classes], trainable=True)
    specific_sms = tf.einsum("ij,jkl->ikl", c_wts, sms)
    common_sm = tf.einsum("j,jkl->kl", common_cwt, sms)
    specific_bias = tf.einsum("ij,jl->il", c_wts, sm_biases)
    common_bias = tf.einsum("j,jl->l", common_cwt, sm_biases)
    
    logits1 = tf.einsum("ik,ikl->il", net, specific_sms) + specific_bias
    logits2 = tf.matmul(net, common_sm) + common_bias
    
    return logits1, logits2
  
def inference_bottleneckv2(net, domain_placeholder, num_classes, num_domains=5, scope="LeNet", emb_dim=2):  
  batch_size = tf.shape(net)[0]
  EMB_SIZE = net.get_shape()[-1]
  with tf.variable_scope(scope, 'mos'):
    DIM = emb_dim
    common_wt = tf.get_variable("common_wt", shape=[1], trainable=False, initializer=tf.ones_initializer)
    specialized_common_wt = tf.get_variable("specialized_wt", shape=[1], initializer=tf.random_normal_initializer(.5, 1e-2))
    emb_matrix = tf.get_variable("emb_matrix", shape=[num_domains, DIM-1], initializer=tf.random_normal_initializer(0, 1e-4))
    
    common_cwt = tf.identity(tf.concat([common_wt, tf.zeros([DIM-1])], axis=0), name='common_cwt')
    
    # Batch size x DIM
    c_wts = tf.nn.embedding_lookup(emb_matrix, domain_placeholder)
    c_wts = tf.concat([tf.ones([batch_size, 1])*specialized_common_wt, c_wts], axis=1)
    # c_wts = tf.sigmoid(c_wts)
    c_wts = tf.reshape(c_wts, [batch_size, DIM])
    
    sms = tf.get_variable("sm_matrices", shape=[DIM, EMB_SIZE, num_classes], trainable=True, initializer=tf.random_normal_initializer(0, 0.05))
    sm_biases = tf.get_variable("sm_bias", shape=[DIM, num_classes], trainable=True)
    specific_sms = tf.einsum("ij,jkl->ikl", c_wts, sms)
    common_sm = tf.einsum("j,jkl->kl", common_cwt, sms)
    specific_bias = tf.einsum("ij,jl->il", c_wts, sm_biases)
    common_bias = tf.einsum("j,jl->l", common_cwt, sm_biases)
    
    diag_tensor = tf.eye(DIM, batch_shape=[num_classes])
    cps = tf.stack([tf.matmul(sms[:, :, _], sms[:, :, _], transpose_b=True) for _ in range(num_classes)])
    orthn_loss = tf.reduce_mean((cps - diag_tensor)**2)
    reg_loss = orthn_loss
    
    logits1 = tf.einsum("ik,ikl->il", net, specific_sms) + specific_bias
    logits2 = tf.matmul(net, common_sm) + common_bias
    
    return logits1, logits2, reg_loss, common_cwt, specialized_common_wt, emb_matrix

def cg(images, label_placeholder, domain_placeholder, num_classes, image_size, num_domains=5, is_training=True, network='bigger_motian', FLAGS=None):
    batch_size = tf.shape(images)[0]

    s = FLAGS.cg_eps
    alpha = 0.5
    
    LN = tf.keras.layers.LayerNormalization(axis=1)
    with tf.variable_scope('crossgrad'):
      label_net_fn = lambda _: LN(get_reprs(_, scope='label_net', image_size=image_size, network=network, is_training=is_training))
      style_net_fn = lambda _: LN(get_reprs(_, scope='style_net', image_size=image_size, network=network, is_training=is_training))
    
      label_logits_fn = lambda _: tf.layers.dense(_, num_classes, name='dense_label', kernel_initializer=tf.random_normal_initializer(0, 0.05))
      style_logits_fn = lambda _: tf.layers.dense(_, num_domains, name='dense_style')
      
      label_net = label_net_fn(images)
      style_net = style_net_fn(images)
      label_logits = label_logits_fn(label_net)
      style_logits = style_logits_fn(style_net)
    
    with tf.variable_scope('crossgrad', reuse=True):
      style_net_stop = tf.stop_gradient(style_net)
      style_probs = tf.nn.softmax(style_logits, axis=-1)

      label_loss = loss(labels = label_placeholder, logits = label_logits, num_classes=num_classes)
      style_loss = loss(labels=domain_placeholder, logits=style_logits, num_classes=num_domains)

      sg = tf.gradients(style_loss, [images])[0]        
      lg = tf.gradients(label_loss, [images])[0]
      
      delJS_x = tf.clip_by_value(sg, clip_value_min=-.1, clip_value_max=.1)
      delJL_x = tf.clip_by_value(lg, clip_value_min=-.1, clip_value_max=.1)

#       delJS_x = tf.Print(delJS_x, data=[tf.shape(delJS_x), tf.shape(delJL_x), tf.shape(images), delJS_x], summarize=10)
      
      lnet  = label_net_fn(images + s*tf.stop_gradient(delJS_x))
      snet = style_net_fn(images + s*tf.stop_gradient(delJL_x))
      lnet_adv = label_net_fn(images + s*tf.stop_gradient(delJL_x))
      snet_adv = style_net_fn(images + s*tf.stop_gradient(delJS_x))

      logit_label_perturb = label_logits_fn(lnet)
      logit_label_perturb_adv = label_logits_fn(lnet_adv)
      logit_style_perturb = style_logits_fn(snet)
      logit_style_perturb_adv = style_logits_fn(snet_adv)

    label_perturb_loss = loss(labels=label_placeholder, logits=logit_label_perturb, num_classes=num_classes)
    label_perturb_loss_adv = loss(labels=label_placeholder, logits=logit_label_perturb_adv, num_classes=num_classes)
    
    style_perturb_loss = loss(logits=logit_style_perturb, labels=domain_placeholder, num_classes=num_domains)
    style_perturb_loss_adv = loss(logits=logit_style_perturb_adv, labels=domain_placeholder, num_classes=num_domains)

    global_step = slim.get_or_create_global_step()

    perturb_norm = tf.identity(label_perturb_loss + style_perturb_loss + label_perturb_loss_adv + style_perturb_loss_adv, name='Perturb_loss')

    final_loss = (alpha * perturb_norm) +  (label_loss + style_loss)
    return final_loss, label_logits, None

def loss(logits, labels, num_classes, clipped=False):
  labels = tf.cast(labels, dtype=tf.int64)
  if not clipped:
    oh_labels = tf.one_hot(labels, num_classes)
    _log_sum = tf.expand_dims(tf.reduce_logsumexp(logits, axis=1), axis=-1)
    _loss = -tf.reduce_sum(oh_labels*(logits - _log_sum), axis=1)
    return tf.reduce_mean(_loss)
  
  else:
    losses = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=tf.losses.Reduction.NONE)
    losses = tf.clip_by_value(losses, 0, 1)
    return tf.reduce_mean(losses)

def training(loss, learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  labels = tf.cast(labels, tf.int64)
  return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.int32))


if __name__ == '__main__':
  train, in_dev, dev, test = prepare_data("data/hpl-devnagari-iso-char-offline")
  
