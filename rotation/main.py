from ResNet import ResNet
import argparse
from utils import *
import numpy as np
import os


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='mnist', help='[cifar10, cifar100, mnist, fashion-mnist, tiny')

    parser.add_argument('--epoch', type=int, default=82, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=200, help='The size of batch per gpu')
    parser.add_argument('--res_n', type=int, default=18, help='4, 18, 34, 50, 101, 152')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--classifier', type=str, default='simple',
                        help='Classifier type: simple/mos')
    parser.add_argument('--L', type=int, default=2,
                        help='L for mos')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--dset_size', type=int, default=1000,
                        help='Dataset size')
    parser.add_argument('--num_domains', type=int, default=20,
                        help='Number of train domains')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    args.checkpoint_dir = args.checkpoint_dir + ("/dset=%s_classifier=%s_L=%d" % (args.dataset, args.classifier, args.L))
    # open session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    
    in_accs, test_accs = [], []
    for seed in range(3):
      os.system("rm -R %s" % args.checkpoint_dir)
      
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
      args.seed = seed
      
      with tf.Session(config=config) as sess:
          cnn = ResNet(sess, args)

          # build graph
          cnn.build_model()

          # show network architecture
          show_all_variables()

          if args.phase == 'train' :
              # launch the graph in a session
              cnn.train()

              print(" [*] Training finished! \n")

              indom_acc = cnn.test()
              test_acc = cnn.test(full=True)
              in_accs.append(indom_acc)
              test_accs.append(test_acc)
              print(" [*] Test finished!")

          if args.phase == 'test' :
              cnn.test()
              print(" [*] Test finished!")
    print ("In-domain accs: %s" % in_accs)
    print ("Test domain accs %s" % test_accs)
    print ("Val and test accs: %0.4f (%0.4f) %0.4f (%0.4f)" % (np.mean(in_accs), np.std(in_accs), np.mean(test_accs), np.std(test_accs)))


if __name__ == '__main__':
    main()