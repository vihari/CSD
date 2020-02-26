from scipy import misc
import tqdm
import pickle
import os

import numpy as np

def eprint(*args):
  _str = " ".join([str(arg) for arg in args])
  sys.stderr.write("%s\n" % _str)

def load_english_hnd():
  # image names are of the form: data/English/Hnd/Img/Sample001/img001-001.png
  fldr = "data/English/Hnd/Img"
  NUM_CLASSES = 59
  NUM_USERS = 55
  IMAGE_SIZE = 32
  
  images, labels, uids = [], [], []
  width, height = IMAGE_SIZE, IMAGE_SIZE
  
  MAX_NUM_DOMAINS = NUM_USERS
  uid = 0
  cache_fname = 'data/english_hnd.pkl'
  if os.path.exists(cache_fname):
    images, labels, uids = pickle.load(open(cache_fname, "rb"))
  else:
    for label in tqdm.tqdm(range(NUM_CLASSES)):
      label_fldr = "%s/Sample%03d" % (fldr, label+1)
      if not os.path.exists(label_fldr):
        continue
      for fname in os.listdir(label_fldr):
        uid = int(fname.split('-')[1][:-4]) - 1

        img = misc.imread(label_fldr + "/" + fname, flatten=True)
        img = misc.imresize(img, (height, width))
        img = img.astype(np.float32)
        img = misc.bytescale(img)
        img = img.astype(np.uint8)   
        
        assert np.max(img) <= 255 and np.min(img) >= 0, "Max and min of image: %f %f" % (np.max(img), np.min(img))
        img = (img-128.)/128.
        assert np.max(img) != np.min(img)
        images.append(img)
        labels.append(label)
        uids.append(uid)
    pickle.dump((images, labels, uids), open(cache_fname, "wb"))
  
  print ("Labels: %s uids: %s" % (labels[:10], uids[:10]))
  print ("Labels: %s uids: %s" % (labels[-10:], uids[-10:]))
  print ("Test images: ", np.max(images[0]), np.min(images[0]))
  
  print ("Read %d examples" % len(images))
  images, labels, uids = np.array(images), np.array(labels), np.array(uids)
  test_idxs = np.where(uids >= NUM_USERS - 15)
  train_idxs = np.where(uids <= NUM_USERS - 25)
  dev_idxs = np.intersect1d(np.where(uids > NUM_USERS - 25), np.where(uids < NUM_USERS - 15))
  train = (images[train_idxs], labels[train_idxs], uids[train_idxs])
  dev = (images[dev_idxs], labels[dev_idxs], uids[dev_idxs])
  test = (images[test_idxs], labels[test_idxs], uids[test_idxs])
  
  return (train, dev, dev, test)


def load_english_fnt():
  # image names are of the form: data/English/Fnt/Img/Sample001/img001-00078.png
  fldr = "data/English/Fnt"
  NUM_CLASSES = 62
  NUM_USERS = 1016
  IMAGE_SIZE = 32
  
  images, labels, uids = [], [], []
  width, height = IMAGE_SIZE, IMAGE_SIZE
  
  MAX_NUM_DOMAINS = NUM_USERS
  uid = 0
  cache_fname = 'data/english_fnt.pkl'
  if os.path.exists(cache_fname):
    images, labels, uids = pickle.load(open(cache_fname, "rb"))
  else:
    for label in tqdm.tqdm(range(NUM_CLASSES)):
      label_fldr = "%s/Sample%03d" % (fldr, label + 1)
      if not os.path.exists(label_fldr):
        continue
      for fname in os.listdir(label_fldr):
        uid = int(fname.split('-')[1][:-4]) - 1

        img = misc.imread(label_fldr + "/" + fname, flatten=True)
        img = misc.imresize(img, (height, width))
        img = img.astype(np.float32)
        img = misc.bytescale(img)
        img = img.astype(np.uint8)   
        
        assert np.max(img) <= 255 and np.min(img) >= 0, "Max and min of image: %f %f" % (np.max(img), np.min(img))
        img = (img-128.)/128.
        assert np.max(img) != np.min(img)
        images.append(img)
        labels.append(label)
        uids.append(uid)
    pickle.dump((images, labels, uids), open(cache_fname, "wb"))
  
  print ("Labels: %s uids: %s" % (labels[:10], uids[:10]))
  print ("Labels: %s uids: %s" % (labels[-10:], uids[-10:]))
  print ("Test images: ", np.max(images[0]), np.min(images[0]))
  
  print ("Read %d examples" % len(images))
  images, labels, uids = np.array(images), np.array(labels), np.array(uids)
  test_idxs = np.where(uids >= NUM_USERS - 100)
  train_idxs = np.where(uids <= NUM_USERS - 500)
  dev_idxs = np.intersect1d(np.where(uids > NUM_USERS - 200), np.where(uids < NUM_USERS - 100))
  train = (images[train_idxs], labels[train_idxs], uids[train_idxs])
  dev = (images[dev_idxs], labels[dev_idxs], uids[dev_idxs])
  test = (images[test_idxs], labels[test_idxs], uids[test_idxs])
  print ("# train, dev, test: %s %s %s" % (np.shape(train[0]), np.shape(dev[0]), np.shape(test[0])))
  
  return (train, dev, dev, test)

def load_font_images():
  # image names are of the form: 32x32/<class name>/<Font_name>.png
  fldr = "../not_notMNIST/32x32/"
  files = os.listdir(fldr)
  
    
  NUM_CLASSES = 62
  NUM_USERS = 1016
  IMAGE_SIZE = 32
  
  images, labels, uids = [], [], []
  width, height = IMAGE_SIZE, IMAGE_SIZE
  
  MAX_NUM_DOMAINS = NUM_USERS
  uid = 0
  cache_fname = 'data/english_fnt.pkl'
  if os.path.exists(cache_fname):
    images, labels, uids = pickle.load(open(cache_fname, "rb"))
  else:
    for label in tqdm.tqdm(range(NUM_CLASSES)):
      label_fldr = "%s/Sample%03d" % (fldr, label + 1)
      if not os.path.exists(label_fldr):
        continue
      for fname in os.listdir(label_fldr):
        uid = int(fname.split('-')[1][:-4]) - 1

        img = misc.imread(label_fldr + "/" + fname, flatten=True)
        img = misc.imresize(img, (height, width))
        img = img.astype(np.float32)
        img = misc.bytescale(img)
        img = img.astype(np.uint8)   
        
        assert np.max(img) <= 255 and np.min(img) >= 0, "Max and min of image: %f %f" % (np.max(img), np.min(img))
        img = (img-128.)/128.
        assert np.max(img) != np.min(img)
        images.append(img)
        labels.append(label)
        uids.append(uid)
    pickle.dump((images, labels, uids), open(cache_fname, "wb"))
  
  print ("Labels: %s uids: %s" % (labels[:10], uids[:10]))
  print ("Labels: %s uids: %s" % (labels[-10:], uids[-10:]))
  print ("Test images: ", np.max(images[0]), np.min(images[0]))
  
  print ("Read %d examples" % len(images))
  images, labels, uids = np.array(images), np.array(labels), np.array(uids)
  test_idxs = np.where(uids >= NUM_USERS - 100)
  train_idxs = np.where(uids <= NUM_USERS - 500)
  dev_idxs = np.intersect1d(np.where(uids > NUM_USERS - 200), np.where(uids < NUM_USERS - 100))
  train = (images[train_idxs], labels[train_idxs], uids[train_idxs])
  dev = (images[dev_idxs], labels[dev_idxs], uids[dev_idxs])
  test = (images[test_idxs], labels[test_idxs], uids[test_idxs])
  print ("# train, dev, test: %s %s %s" % (np.shape(train[0]), np.shape(dev[0]), np.shape(test[0])))
  
  return (train, dev, dev, test)

def test_enghnd():
  train, _, dev, test = load_english_hnd()
  print ("Train max and min: %d %d" % (np.max(train[-1]), np.min(train[-1])))
  print ("Dev max and min: %d %d" % (np.max(dev[-1]), np.min(dev[-1])))
  print ("Test max and min: %d %d" % (np.max(test[-1]), np.min(test[-1])))
  print (np.shape(train[0]))
  
def test_engfnt():
  train, _, dev, test = load_english_fnt()
  print ("Train max and min: %d %d" % (np.max(train[-1]), np.min(train[-1])))
  print ("Dev max and min: %d %d" % (np.max(dev[-1]), np.min(dev[-1])))
  print ("Test max and min: %d %d" % (np.max(test[-1]), np.min(test[-1])))
  print (np.shape(train[0]))

if __name__ == '__main__':
#   test_enghnd()
  test_engfnt()