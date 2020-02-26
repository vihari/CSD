import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random
from scipy import misc

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

import sys
  
def eprint(*args):
  _str = " ".join([str(arg) for arg in args])
  sys.stderr.write("%s\n" % _str)
  
def _load_lipitk(fldr):
    images, labels, uids = [], [], []

    IMAGE_SIZE = 32
    width, height = IMAGE_SIZE, IMAGE_SIZE
    MAX_NUM_DOMAINS = 110
    uid = 0
    for dom in range(MAX_NUM_DOMAINS):
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
        img = img/255.
        images.append(img)
        labels.append(li)
        uids.append(uid)
      uid += 1

    print ("Labels: %s uids: %s" % (labels[:10], uids[:10]))
    print ("Labels: %s uids: %s" % (labels[-10:], uids[-10:]))
    print ("Test images: ", np.max(images[0]), np.min(images[0]))

    print ("Read %d examples" % len(images))
    return np.array(images), np.array(labels), np.array(uids)


def load_lipitk(num_train):
  data_folder = "../ucg_stack/data/hpl-devnagari-iso-char-offline"
  np.random.seed(1)

  imgs, labels, uids = _load_lipitk(data_folder)
  
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
  uid_to_idx = {uuids[_]:_ for _ in range(len(uuids))}
  
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
  in_dev = (train[0][train_sp:], train[1][train_sp:], train[2][train_sp:])
  train = (train[0][:train_sp], train[1][:train_sp], train[2][:train_sp])
  print (np.shape(train[0]), np.shape(in_dev[0]), np.shape(dev[0]), np.shape(test[0]))
  split_dats = [train, in_dev, dev, test]
  
  eprint ("Number of train, dev and test domains: %d %d %d" % (len(train_domains), len(dev_domains), len(test_domains)))
  eprint ("Number of examples per split: %d %d %d %d" % (len(split_dats[0][0]), len(split_dats[1][0]), len(split_dats[2][0]), len(split_dats[3][0])))
  eprint ("Train labels: ", np.unique(split_dats[0][1]))
  train_y, test_y = np.zeros([len(train[0]), 111]), np.zeros([len(test[0]), 111])
  train_y[np.arange(len(train_y)), train[1]] = 1
  test_y[np.arange(len(test_y)), test[1]] = 1
  
  return np.expand_dims(train[0], -1), train_y, train[2], np.expand_dims(test[0], -1), test_y, test[2]


def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0
    
    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_cifar100() :
    (train_data, train_labels), (test_data, test_labels) = cifar100.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0
    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 100)
    test_labels = to_categorical(test_labels, 100)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_fashion() :
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_tiny() :
    IMAGENET_MEAN = [123.68, 116.78, 103.94]
    path = './tiny-imagenet-200'
    num_classes = 200

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype=np.float32)
    y_train = np.zeros([num_classes * 500], dtype=np.float32)

    trainPath = path + '/train'

    print('loading training images...')

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = misc.imread(os.path.join(sChildPath, c), mode='RGB')
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype=np.float32)
    y_test = np.zeros([num_classes * 50], dtype=np.float32)

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = misc.imread(sChildPath, mode='RGB')
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images : ' + str(i))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # X_train /= 255.0
    # X_test /= 255.0

    # for i in range(3) :
    #     X_train[:, :, :, i] =  X_train[:, :, :, i] - IMAGENET_MEAN[i]
    #     X_test[:, :, :, i] = X_test[:, :, :, i] - IMAGENET_MEAN[i]

    X_train, X_test = normalize(X_train, X_test)


    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    X_train = np.transpose(X_train, [0, 3, 2, 1])
    X_test = np.transpose(X_test, [0, 3, 2, 1])

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    return X_train, y_train, X_test, y_test

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def get_annotations_map():
    valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch, img_size, dataset_name):
    if dataset_name == 'mnist' :
        batch = _random_crop(batch, [img_size, img_size], 4)

    elif dataset_name =='tiny' :
        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [img_size, img_size], 8)

    else :
        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [img_size, img_size], 4)
    return batch