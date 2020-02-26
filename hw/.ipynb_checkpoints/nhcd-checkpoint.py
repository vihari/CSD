import numpy as np
import os
import sys

from matplotlib.pyplot import imread
import tensorflow as tf
import lipitk

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

def set_num_classes(data_fldr):
  global NUM_CLASSES
  if data_fldr.find('numerals') >= 0: 
    NUM_CLASSES = 10
  elif data_fldr.find('consonants') >= 0:
    NUM_CLASSES = 36
  elif data_fldr.find('vowels') >= 0:
    NUM_CLASSES = 12
  else:
    raise AssertionError("Unrecognized data folder!")
  return NUM_CLASSES


task_type = "consonants"
DATA_FLDR = "data/nhcd/nhcd/" + task_type
NUM_CLASSES = set_num_classes(DATA_FLDR)

"""
Helper to load lipitk data and network definition to process it.
"""

def eprint(*args):
  _str = " ".join([str(arg) for arg in args])
  sys.stderr.write("%s\n" % _str)


def load_data(fldr, doms):
  global NUM_CLASSES
  images, labels, uids = [], [], []
  sdoms = set(doms)
  
  for li in range(NUM_CLASSES):
    # vowels and consonant folder label naming starts with 1 and not 0
    if fldr.find('numeral') < 0:
      label_name = str(li + 1)
    else:
      label_name = str(li)
    label_fldr = os.path.join(fldr, label_name)
    for fname in os.listdir(label_fldr):
      uid = fname.split('_')[0]
      if uid in doms:
        img = imread(label_fldr + "/" + fname)
        assert np.max(img) <= 255 and np.min(img) >= 0, "Max and min of image: %f %f" % (np.max(img), np.min(img))
        img = (img - 128.)/128.
        images.append(img)
        labels.append(li)
        uids.append(uid)
  
  print ("Read %d examples, all images are in range: %f, %f" % (len(images), np.min(images), np.max(images)))
  return np.array(images), np.array(labels), np.array(uids)


def prepare_data():
  np.random.seed(1)

  domain_ids = []
  data_folder = DATA_FLDR
  with open(data_folder + "/all_domains.txt") as f:
    for l in f:
      l = l.strip()
      domain_ids.append(l)
    
  train_domains = domain_ids[:27]
  dev_domains = domain_ids[27:32]
  test_domains = domain_ids[32:]
  
  imgs, labels, uids = load_data(data_folder, domain_ids)
  train_data, dev_data, test_data = [], [], []
  uuids = np.unique(uids)
  uid_to_idx = {domain_ids[_]:_ for _ in range(len(domain_ids))}
#   uid_to_idx = {uuids[_]: _ for _ in range(len(uuids))}
  
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
  in_dev = dev
  print (np.shape(train[0]), np.shape(in_dev[0]), np.shape(dev[0]), np.shape(test[0]))
  split_dats = [train, in_dev, dev, test]
  
  eprint ("Number of train, dev and test domains: %d %d %d" % (len(train_domains), len(dev_domains), len(test_domains)))
  eprint ("Number of examples per split: %d %d %d %d" % (len(split_dats[0][0]), len(split_dats[1][0]), len(split_dats[2][0]), len(split_dats[3][0])))
  eprint ("Train labels: ", np.unique(split_dats[0][1]))
  return split_dats

def training(loss, learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

if __name__ == '__main__':
  train, in_dev, dev, test = prepare_data("data/nhcd/nhcd/vowels")
  