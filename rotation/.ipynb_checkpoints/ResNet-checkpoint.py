import time
from ops import *
from utils import *
import numpy as np
import tqdm
from scipy.ndimage import rotate as rot
import pickle

def custom_rot(img, angle, reshape=False):
  _rot = rot(img, angle, reshape=reshape)
  mx, mn = 1, 0
  return (_rot - mn)/(mx - mn)

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200
            
        if self.dataset_name == 'lipitk' :
            self.train_x, self.train_y, self.train_u, self.test_x, self.test_y, self.test_u = load_lipitk(args.num_domains)
            self.img_size = 32
            self.c_dim = 1
            self.label_dim = 111

        fname = "cache2000/%s.pkl" % self.dataset_name
        self.num_domains = 10
        ROTATE = True
      
        if not ROTATE:
          self.train_u = np.random.randint(0, self.num_domains, [len(self.train_x)])
          self.ind_test_x, self.ind_test_y = self.test_x, self.test_y
          self.full_test_x, self.full_test_y = self.test_x, self.test_y
          self.ind_test_u = np.random.randint(0, self.num_domains, [len(self.ind_test_x)])
          self.full_test_u = np.random.randint(0, self.num_domains, [len(self.full_test_x)])
        else:
          self.num_domains = 5
          if os.path.exists(fname):
            with open(fname, 'rb') as f:
              self.train_x, self.train_y, self.train_u, self.test_x, self.test_y, self.test_u = pickle.load(f)
          else:
            if self.dataset_name == 'mnist':
              self.train_x, self.train_y = self.train_x[:2000], self.train_y[:2000]
            else:
              self.train_x, self.train_y = self.train_x[:10000], self.train_y[:10000]

            all_train_x, all_train_y, all_train_u = [], [], []
            all_test_x, all_test_y, all_test_u = [], [], []
            for idx, angle in tqdm.tqdm(enumerate(range(15, 90, 15)), desc='Rotating train images'):
              for train_image in self.train_x:
                _rot = custom_rot(train_image, angle, reshape=False)
                all_train_x.append(_rot)
              all_train_y += self.train_y.tolist()
              all_train_u += [idx]*len(self.train_y)
              print (len(all_train_x), len(all_train_y), len(all_train_u))
            train_x = np.array(all_train_x)
            train_y = np.array(all_train_y)
            train_u = np.array(all_train_u)

            shuffle_idxs = np.random.permutation(len(train_x))
            train_x, train_y, train_u = train_x[shuffle_idxs], train_y[shuffle_idxs], train_u[shuffle_idxs]
            self.original_train_u = train_u

            print ("Rotating %d images for in-domain" % len(self.test_x))
            for ai, angle in tqdm.tqdm(enumerate(range(15, 90, 15)), desc='Rotating test images'):
              for test_image in self.test_x:
                all_test_x.append(custom_rot(test_image, angle, reshape=False))
              all_test_y += self.test_y.tolist()
              all_test_u += [ai]*len(self.test_y)
            self.ind_test_x = np.array(all_test_x)
            self.ind_test_y = np.array(all_test_y)
            self.ind_test_u = np.array(all_test_u)

          print ("Rotating %d images" % len(self.test_x))
          all_test_x, all_test_y, all_test_u = [], [], []
          angles = [0, 90]
          for angle in tqdm.tqdm(angles, desc='Rotating test images'):
            for test_image in self.test_x:
              all_test_x.append(custom_rot(test_image, angle, reshape=False))
            all_test_y += self.test_y.tolist()
            all_test_u += [0]*len(self.test_y)
          self.full_test_x = np.array(all_test_x)
          self.full_test_y = np.array(all_test_y)
          self.full_test_u = np.array(all_test_u)
          
          self.train_x, self.train_y, self.train_u = train_x, train_y, train_u
          print ("Shapes: %s %s %s" % (np.shape(self.train_x), np.shape(self.train_y), np.shape(self.train_u)))
          
        self.test_x, self.test_y, self.test_u = self.ind_test_x, self.ind_test_y, self.ind_test_u
        
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr
        self.args = args


    ##################################################################################
    # Generator
    ##################################################################################
    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

                
            if len(residual_list) > 1:
                ########################################################################################################
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

                for i in range(1, residual_list[1]) :
                    x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

                ########################################################################################################
              
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

                for i in range(1, residual_list[2]) :
                    x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

                ########################################################################################################

                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

                for i in range(1, residual_list[3]) :
                    x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

                ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=128, scope='logit')
            x = tf.keras.layers.LayerNormalization(axis=1)(x)
            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')
        self.train_domains = tf.placeholder(tf.int32, [self.batch_size], name='train_domains')

        self.test_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='test_labels')
        self.test_domains = tf.placeholder(tf.int32, [self.batch_size], name='test_domains')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_reprs = self.network(self.train_inptus)
        self.test_reprs = self.network(self.test_inptus, is_training=False, reuse=True)

        if self.args.classifier == 'mos':
          with tf.variable_scope(''):
            self.train_loss, self.train_accuracy = mos_regression_lossv2(reprs=self.train_reprs, label=self.train_labels, domain=self.train_domains, num_domains=self.num_domains, L=self.args.L)
          with tf.variable_scope('', reuse=True):
            self.test_loss, self.test_accuracy = mos_regression_lossv2(reprs=self.test_reprs, label=self.test_labels, domain=self.test_domains, num_domains=self.num_domains, L=self.args.L)
        else: 
          with tf.variable_scope(''):
            self.train_loss, self.train_accuracy = regression_loss(reprs=self.train_reprs, label=self.train_labels, domain=self.train_domains, num_domains=self.num_domains)
          with tf.variable_scope('', reuse=True):
            self.test_loss, self.test_accuracy = regression_loss(reprs=self.test_reprs, label=self.test_labels, domain=self.test_domains, num_domains=self.num_domains)
        
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss

        """ Training """
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * .1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                idxs = np.random.choice(np.arange(len(self.train_x)), self.batch_size)
                batch_x = self.train_x[idxs]
                batch_y = self.train_y[idxs]
                batch_u = self.train_u[idxs]

                # uncomment if do not wish to augment
                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)
                
                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.train_domains: batch_u,
                    self.lr : epoch_lr
                }

                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                dvars = [var for var in tf.trainable_variables() if var.name.find('common_specialized_wt')>=0 or var.name.find('emb_mat')>=0]
                if idx == self.iteration - 1:
                  test_loss, test_accuracy = 0, 0
                  nsteps = len(self.test_x)//self.batch_size
                  for _ in range(nsteps):
                    batch_test_x = self.test_x[_*self.batch_size: (_+1)*self.batch_size]
                    batch_test_y = self.test_y[_*self.batch_size: (_+1)*self.batch_size]
                    batch_test_u = self.test_u[_*self.batch_size: (_+1)*self.batch_size]

                    test_feed_dict = {
                        self.test_inptus : batch_test_x,
                        self.test_labels : batch_test_y,
                        self.test_domains: batch_test_u
                    }
                    # test
                    summary_str, _test_loss, _test_accuracy = self.sess.run(
                        [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                    test_loss += _test_loss
                    test_accuracy += _test_accuracy

                  test_loss /= nsteps
                  test_accuracy /= nsteps
                  self.writer.add_summary(summary_str, counter)  

                  # display training status
                  counter += 1
                  print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f loss: %.2f" \
                        % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr, train_loss))
                  
                  print ("Debug vars: ", self.sess.run(dvars))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        
        # save model for final step
        self.save(self.checkpoint_dir, counter)
        if self.args.classifier == 'mos':
          print ("EMB mat: ", self.sess.run(dvars))
          classifier = [var for var in tf.trainable_variables() if var.name.find('sm_matrices')>=0][0]
          np_classifier = self.sess.run(classifier)
          train_x, train_y, train_u = self.train_x, self.train_y, self.original_train_u
          nsteps = len(train_x)//self.batch_size
          np_train_reprs = []
          for _ in range(nsteps):
            idxs = range(_*self.batch_size, (_+1)*self.batch_size)
            batch_x, batch_y, batch_u = train_x[idxs], train_y[idxs], train_u[idxs]
            feed_dict = {
              self.test_inptus : batch_x,
              self.test_labels : batch_y,
              self.test_domains: batch_u,
            }
            _train_reprs = self.sess.run(self.test_reprs, feed_dict=feed_dict)
            np_train_reprs += _train_reprs.tolist()
          np_train_reprs = np.array(np_train_reprs)
          with open("logs/dataset=%s_seed=%d_supervised-debug.pkl" % (self.dataset_name, self.args.seed), "wb") as f:
            pickle.dump([np_train_reprs, train_y, train_u, np_classifier, train_u], f)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self, full=False):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if full:
          test_x, test_y, test_u = self.full_test_x, self.full_test_y, self.full_test_u
        else:
          test_x, test_y, test_u = self.test_x, self.test_y, self.test_u
        
        nsteps = len(test_x)//self.batch_size
        test_accuracy, test_loss = 0, 0
        for _ in range(nsteps):
          batch_test_x = test_x[_*self.batch_size: (_+1)*self.batch_size]
          batch_test_y = test_y[_*self.batch_size: (_+1)*self.batch_size]
          batch_test_u = test_u[_*self.batch_size: (_+1)*self.batch_size]

          test_feed_dict = {
              self.test_inptus : batch_test_x,
              self.test_labels : batch_test_y,
              self.test_domains: batch_test_u
          }
          # test
          summary_str, _test_loss, _test_accuracy = self.sess.run(
              [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
          test_loss += _test_loss
          test_accuracy += _test_accuracy

        test_loss /= nsteps
        test_accuracy /= nsteps

        print("test_accuracy: {}".format(test_accuracy))
        return test_accuracy
