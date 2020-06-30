import tensorflow as tf
import tensorflow.contrib as tf_contrib


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init


def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 4 :
        x = [1]
    
    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

##################################################################################
# Loss function
##################################################################################

def regression_loss(reprs, label, domain, num_domains):
  _, accuracy, all_losses = mos_regression_lossv2(reprs, label, domain, num_domains, debug=True)
  return all_losses[1], accuracy

def regression_loss2(reprs, label, reuse=False, normalize=False) :
  num_classes = label.get_shape()[-1]

  logit = tf.layers.dense(reprs, num_classes, name='softmax_layer', reuse=reuse)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
  prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
  accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
  return loss, accuracy
  
def mos_regression_loss(reprs, label, domain, num_domains):
  DIM = 2
  batch_size = tf.shape(reprs)[0]
  EMB_SIZE = 128
  num_classes = label.get_shape()[-1]
  
  emb_matrix = tf.get_variable("emb_matrix", shape=[num_domains, DIM], initializer=tf.random_normal_initializer)
  common_var = tf.get_variable("common_var", shape=[DIM], initializer=tf.zeros_initializer)

  common_cwt = tf.sigmoid(common_var)
  common_cwt /= tf.norm(common_cwt)
  emb_matrix = tf.sigmoid(emb_matrix)
  emb_matrix /= tf.expand_dims(tf.norm(emb_matrix, axis=1), 1)

  # Batch size x DIM
  c_wts = tf.nn.embedding_lookup(emb_matrix, domain)
  c_wts = tf.reshape(c_wts, [batch_size, DIM])

  sms = tf.get_variable("sm_matrices", shape=[DIM, EMB_SIZE, num_classes], trainable=True)
  biases = tf.get_variable("sm_bias", shape=[DIM, num_classes], trainable=True)
  
  specific_sms = tf.einsum("ij,jkl->ikl", c_wts, sms)
  specific_bias = tf.einsum("ij,jl->il", c_wts, biases)
  common_sm = tf.einsum("j,jkl->kl", common_cwt, sms)
  common_bias = tf.einsum("j,jl->l", common_cwt, biases)
  
  logits1 = tf.einsum("ik,ikl->il", reprs, specific_sms) + specific_bias
  logits2 = tf.matmul(reprs, common_sm) + common_bias
  
  loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits1))
  loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits2))

  loss = 0.5*loss1 + 0.5*loss2
  predictions = tf.equal(tf.argmax(logits2, axis=-1), tf.argmax(label, axis=1))
  accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

  return loss, accuracy

def mos_regression_lossv2(reprs, label, domain, num_domains, L=2, cs_wt=0, debug=False):
  DIM = L
  batch_size = tf.shape(reprs)[0]
  EMB_SIZE = 128
  num_classes = label.get_shape()[-1]

  emb_matrix = tf.get_variable("emb_mat", shape=[num_domains, DIM-1], initializer=tf.random_normal_initializer(0, 1e-4))
  common_wt = tf.get_variable("common_wt", shape=[1], initializer=tf.ones_initializer)
  common_specialized_wt = tf.get_variable("common_specialized_wt", shape=[], initializer=tf.random_normal_initializer(cs_wt, 1e-2))
  common_cwt = tf.concat([common_wt, tf.zeros([DIM-1])], axis=0)

  # Batch size x DIM
  c_wts = tf.nn.embedding_lookup(emb_matrix, domain)
  c_wts = tf.concat([tf.ones([batch_size, 1])*common_specialized_wt, c_wts], axis=1)
  c_wts = tf.reshape(c_wts, [batch_size, DIM])

  c_wts = tf.nn.sigmoid(c_wts)
  
  sms = tf.get_variable("sm_matrices", shape=[DIM, EMB_SIZE, num_classes], trainable=True, initializer=tf.random_normal_initializer(0, 0.05))
  biases = tf.get_variable("sm_bias", shape=[DIM, num_classes], trainable=True, initializer=tf.random_normal_initializer(0, 0.05))
  
  specific_sms = tf.einsum("ij,jkl->ikl", c_wts, sms)
  specific_bias = tf.einsum("ij,jl->il", c_wts, biases)
  common_sm = tf.einsum("j,jkl->kl", common_cwt, sms)
  common_bias = tf.einsum("j,jl->l", common_cwt, biases)
  
  logits1 = tf.einsum("ik,ikl->il", reprs, specific_sms) + specific_bias
  logits2 = tf.matmul(reprs, common_sm) + common_bias
  
  loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits1))
  loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits2))

  # C x L x L
  diag_tensor = tf.eye(DIM, batch_shape=[num_classes])
  cps = tf.stack([tf.matmul(sms[:, :, _], sms[:, :, _], transpose_b=True) for _ in range(num_classes)])
  # (1 - diag_tensor) *
  orthn_loss = tf.reduce_mean((cps - diag_tensor)**2)
  
  loss = 0.5*(loss1 + loss2) + orthn_loss
  predictions = tf.equal(tf.argmax(logits2, axis=-1), tf.argmax(label, axis=1))
  accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

  if debug:
    return loss, accuracy, [loss1, loss2, orthn_loss]
  else:
    return loss, accuracy
