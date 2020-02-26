#!/usr/bin/python
import tensorflow as tf
import speech_models as models
from tensorflow.python.framework import ops
import numpy as np


def rankL(np_rank):
    r = int(np_rank[-1])
    _l = 0
    for k in range(1, r+1):
        _l += 1./k
    return np.float32(_l)

"""
labels are assumed to be 1 hot encoded
"""
def warp_loss(labels, logits):
    # for easy broadcasting
    labels, logits = tf.transpose(labels, [1, 0]), tf.transpose(logits, [1, 0])
    # logits -- LxB
    # labels -- LxB
    f_y = tf.reduce_sum(logits*labels, axis=0)
    # logits can get too big that 1+logits - f_y can evaluate to 0 instead of 1, so caution by putting brackets around diff.
    rank = tf.reduce_sum(tf.maximum(tf.sign(1+(logits-f_y)), 0), axis=0)
    # rank = tf.Print(rank, data=[rank], message="Rank, desired, labels, logits", summarize=100)
    diff = tf.reduce_sum(tf.maximum(1+(logits-f_y), 0), axis=0)
    with tf.control_dependencies([tf.assert_greater(rank, tf.zeros_like(rank), data=[tf.transpose(1+(logits-f_y), [1, 0]), f_y, rank], summarize=10000)]):
        return tf.py_func(rankL, [rank], tf.float32) * diff/rank

slim=tf.contrib.slim
def cg_losses (net, style_net, style, label, num_styles, num_labels, inputs, scales, label_net_fn=None, style_net_fn=None):
    net_projector_layer = tf.layers.Dense(units=128, activation=tf.nn.leaky_relu)
    style_net_projector_layer = tf.layers.Dense(units=128, activation=tf.nn.leaky_relu)
    
    def net_projector(_):
      net = net_projector_layer(_)
      return tf.keras.layers.LayerNormalization(axis=1)(net)
    
    def style_net_projector(_):
      net = style_net_projector_layer(_)
      return tf.keras.layers.LayerNormalization(axis=1)(net)
    
    net = net_projector(net)
    style_net = style_net_projector(style_net)

    label_softmax = tf.layers.Dense(units=num_labels, activation=None, use_bias=False, kernel_initializer=tf.random_uniform_initializer(-0.076, 0.076))
    style_softmax = tf.layers.Dense(units=num_styles, activation=None)
    
    style_logits = style_softmax(style_net)
    label_logits = label_softmax(net)
    
    batch_size = tf.shape(inputs)[0]

    # assert (not expert_sm) or (smparam is not None), "Trying to train expert softmax without feeding smparam matrix"
    s = scales["epsilon"]
    alpha = scales["alpha"]
    with tf.variable_scope('crossgrad', reuse=True):            
        label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = label_logits), name='Label_loss')
        style_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = style, logits = style_logits), name='Style_loss')
        
        sg = tf.gradients(style_loss, inputs)[0]        
        lg = tf.gradients(label_loss, inputs)[0]
        delJS_x = sg
        delJL_x = lg
        delJS_x = tf.clip_by_value(sg, clip_value_min=-0.1, clip_value_max=0.1)
        delJL_x = tf.clip_by_value(lg, clip_value_min=-0.1, clip_value_max=0.1)
        #print ("Len is ", len(tf.trainable_variables()))

        lnet  = label_net_fn(inputs + s*tf.stop_gradient(delJS_x))
        snet = style_net_fn(inputs + s*tf.stop_gradient(delJL_x))
        lnet_adv = label_net_fn(inputs + s*tf.stop_gradient(tf.sign(delJL_x)))
        snet_adv = style_net_fn(inputs + s*tf.stop_gradient(delJS_x))

        lnet = net_projector(lnet)
        snet = style_net_projector(snet)
        
    logit_label_perturb = label_softmax(lnet)
    logit_style_perturb = style_softmax(snet)

    label_perturb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit_label_perturb), name='Label_perturb_loss')

#     label_perturb_loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit_label_perturb_adv), name='Label_perturb_loss_adv')

    style_perturb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=style, logits=logit_style_perturb), name='Style_perturb_loss')

    global_step = slim.get_or_create_global_step()

    # style_perturb_loss
    perturb_norm = tf.identity(label_perturb_loss, name='Perturb_loss')
    # perturb_norm = tf.identity(label_perturb_loss_adv)

    tf.losses.add_loss( alpha * perturb_norm)
    tf.summary.scalar("Label_loss", label_loss)
    tf.losses.add_loss( label_loss)
    tf.losses.add_loss( style_loss)
    final_loss = alpha*perturb_norm + (1 - alpha)*label_loss + style_loss
    return final_loss, label_logits

  
def mos (net, style, label, num_labels, num_styles, inputs, FLAGS):
    net = tf.layers.dense(net, units=128, activation=tf.nn.leaky_relu)
    net = tf.keras.layers.LayerNormalization(axis=1)(net)

    batch_size = tf.shape(net)[0]
    emb_size = net.get_shape()[-1]
        
    # style is already one-hot encoded
    num_latent_styles = FLAGS.num_uids
    num_actual_styles = num_styles
    assert num_latent_styles > 0, "Trying to set the mos model without setting the number of uids param" 

    emb_matrix = tf.get_variable("emb_mat", shape=[num_actual_styles, num_latent_styles], initializer=tf.random_normal_initializer, trainable=True)
    common_var = tf.get_variable("common_var", shape=[num_latent_styles], initializer=tf.zeros_initializer)

    common_cwt = tf.nn.sigmoid(common_var)
    common_cwt /= tf.norm(common_cwt)
    emb_matrix = tf.nn.sigmoid(emb_matrix)
    emb_matrix /= tf.expand_dims(tf.norm(emb_matrix, axis=1), 1)
    
    global_step = tf.contrib.framework.get_or_create_global_step()  
    emb_matrix = tf.cond(tf.equal(tf.mod(global_step, 337), 0), false_fn=lambda: emb_matrix, true_fn=lambda: tf.Print(emb_matrix, message="Embs", data=[emb_matrix], summarize=5))
    
    sm_w = tf.get_variable("sm_w", shape=[num_latent_styles, emb_size, num_labels])
    sm_bias = tf.get_variable("sm_bias", shape=[num_latent_styles, num_labels])

    style_embs = tf.nn.embedding_lookup(emb_matrix, tf.argmax(style, axis=1))
    specific_sms = tf.einsum("iu,ujl->ijl", style_embs, sm_w)
    specific_bias = tf.einsum("iu,ul->il", style_embs, sm_bias)
    common_sm = tf.einsum("j,jkl->kl", common_cwt, sm_w)
    common_bias = tf.einsum("j,jl->l", common_cwt, sm_bias)
    
    logits1 = tf.einsum("ij,ijl->il", net, specific_sms)
    logits2 = tf.matmul(net, common_sm)
    
    with tf.variable_scope('crossgrad', reuse=True):
      label_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits1), name='Label_loss_specific')
      label_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits2), name='Label_loss_common')

    final_loss = 0.5 * (label_loss1 + label_loss2)

    tf.summary.scalar("Label_loss", label_loss1)
    tf.losses.add_loss(final_loss)
    
    return final_loss, logits1, logits2

def mos2 (net, style, label, num_labels, num_styles, inputs, FLAGS):
    net = tf.layers.dense(net, units=128, activation=tf.nn.leaky_relu)
    net = tf.keras.layers.LayerNormalization(axis=1)(net)

    batch_size = tf.shape(net)[0]
    emb_size = net.get_shape()[-1]
        
    # style is already one-hot encoded
    num_latent_styles = FLAGS.num_uids
    num_actual_styles = num_styles
    assert num_latent_styles > 0, "Trying to set the mos model without setting the number of uids param" 

    # S x L
    with tf.device('/cpu:0'):
      emb_matrix = tf.get_variable("emb_mat", shape=[num_actual_styles, num_latent_styles-1], initializer=tf.random_normal_initializer(0, 1e-4), trainable=True)
    common_wt = tf.get_variable("common_wt", shape=[1], initializer=tf.ones_initializer, trainable=False)
    common_specialized_wt = tf.get_variable("common_specialized_wt", shape=[], initializer=tf.random_normal_initializer(0, 1e-2))
    common_cwt = tf.concat([common_wt, tf.zeros([num_latent_styles-1])], axis=0)
    
    global_step = tf.contrib.framework.get_or_create_global_step()  
    emb_matrix = tf.cond(tf.equal(tf.mod(global_step, 337), 0), false_fn=lambda: emb_matrix, true_fn=lambda: tf.Print(emb_matrix, message="Embs", data=[emb_matrix, common_specialized_wt, common_cwt], summarize=5))
    
    sm_w = tf.get_variable("sm_w", shape=[num_latent_styles, emb_size, num_labels], initializer=tf.random_uniform_initializer(-0.076, 0.076))

    style_embs = tf.nn.embedding_lookup(emb_matrix, tf.argmax(style, axis=1))
    style_embs = tf.concat([tf.ones([batch_size, 1])*common_specialized_wt, style_embs], axis=1)
    
    style_embs = tf.nn.sigmoid(style_embs)
    
    specific_sms = tf.einsum("iu,ujl->ijl", style_embs, sm_w)
    common_sm = tf.einsum("j,jkl->kl", common_cwt, sm_w)
    
    logits1 = tf.einsum("ij,ijl->il", net, specific_sms)
    logits2 = tf.matmul(net, common_sm)
    # C x L x L
    diag_tensor = tf.eye(num_latent_styles, batch_shape=[num_labels])
    cps = tf.stack([tf.matmul(sm_w[:, :, _], sm_w[:, :, _], transpose_b=True) for _ in range(num_labels)])
    orthn_loss = tf.reduce_mean((cps - diag_tensor)**2)
    
    with tf.variable_scope('crossgrad', reuse=True):
      label_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits1), name='Label_loss_specific')
      label_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits2), name='Label_loss_common')

    lmbda = FLAGS.lmbda
    if lmbda > 0:
      final_loss = lmbda*(label_loss1) + (1-lmbda)*(label_loss2) + orthn_loss
    else:
      final_loss = label_loss2

    tf.summary.scalar("Label_loss", label_loss1)
    tf.losses.add_loss(final_loss)
    
    return final_loss, logits1, logits2
  
def mos_tune_project (net):
  return tf.layers.dense(net, units=128, activation=tf.nn.leaky_relu, reuse=True)

def mos_tune (net, tune_var, label, num_labels, FLAGS):
  """
  tune_var is [num_latent_styles]
  """  
  batch_size = tf.shape(net)[0]
  emb_size = net.get_shape()[-1]

  num_latent_styles = FLAGS.num_uids
  assert num_latent_styles > 0, "Trying to set the mos model without setting the number of uids param" 

  sm_w = tf.get_variable("sm_w", shape=[num_latent_styles, emb_size, num_labels])
  sm_bias = tf.get_variable("sm_bias", shape=[num_latent_styles, num_labels])
  
  specific_sms = tf.einsum("u,ujl->jl", tune_var, sm_w)
  specific_bias = tf.einsum("u,ul->l", tune_var, sm_bias)
  
  logits1 = tf.einsum("ij,jl->il", net, specific_sms)
  logits = logits1 

  with tf.variable_scope('crossgrad', reuse=True):
    label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits), name='Label_loss')

  final_loss = label_loss

  return logits, net

def simple(net, style, label, num_labels, num_styles, inputs, FLAGS):
  FLAGS.lmbda = 0
  final_loss, logits1, logits2 = mos2 (net, style, label, num_labels, num_styles, inputs, FLAGS)
  return final_loss, logits2

def simple2(net, style, label, num_labels, num_styles, inputs, FLAGS, debug=False):
    net = tf.layers.dense(net, units=128, activation=tf.nn.leaky_relu)
    net = tf.keras.layers.LayerNormalization(axis=1)(net)
    
    batch_size = tf.shape(net)[0]
    emb_size = net.get_shape()[-1]
        
    sm_w = tf.get_variable("sm_w", shape=[emb_size, num_labels], initializer=tf.random_normal_initializer(0, 1e-2))
    sm_bias = tf.get_variable("sm_bias", shape=[num_labels])
        
    logits = tf.einsum("ij,jl->il", net, sm_w)
    
    with tf.variable_scope('crossgrad', reuse=True):
        label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits), name='Label_loss')

    final_loss = label_loss

    tf.summary.scalar("Label_loss", label_loss)
    tf.losses.add_loss(final_loss)

    if debug:
        return final_loss, logits, net
    else:
        return final_loss, logits


class ReverseGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "ReverseGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _reverse_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
_reverse_grad = ReverseGradientBuilder()


def dan_loss (net, style, num_styles, label_logits, label):

    global_step = slim.get_or_create_global_step()
    p = tf.cast(tf.maximum(global_step - 1000, 0), tf.float32)/18000.
    l = tf.stop_gradient((2./(1. + tf.exp(-15.*p))) -1)
    l = tf.minimum(l, 0.8)
    net = _reverse_grad(net, l)
    gan_logits = models.descriminator(net, num_styles)

    gan_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = style, logits = gan_logits),name='gan_loss')
 
    label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = label_logits), name='Label_loss')
    #gan_loss = tf.Print(gan_loss, [gan_loss, l], message="GAN Loss: ")
    gan_loss = tf.cond(tf.equal(tf.mod(global_step, 100), 0), lambda: tf.Print(gan_loss, [gan_loss, l], message="GAN Loss: "), lambda: gan_loss)
    tf.losses.add_loss(gan_loss)
    tf.losses.add_loss(label_loss)
    return label_loss
