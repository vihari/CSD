import argparse

import torch
from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.005, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    # keep this for dataloaders.
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18_csd")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default="test", help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--train_all", action='store_true', help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", action='store_true', help="Use nesterov")
    
    return parser.parse_args()

def one_hot(ids, depth):
    z = np.zeros([len(ids), depth])
    z[np.arange(len(ids)), ids] = 1
    return z

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            NUM_DOMAINS = 3
            oh_dids = torch.tensor(one_hot(d_idx, NUM_DOMAINS), dtype=torch.float, device='cuda')
            
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)

            self.optimizer.zero_grad()

            specific_logit, class_logit = self.model(data, oh_dids) 
            specific_loss = criterion(specific_logit, class_l)

            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)

            sms = self.model.sms
            K = 2
            diag_tensor = torch.stack([torch.eye(K) for _ in range(self.n_classes)], dim=0).cuda()
            cps = torch.stack([torch.matmul(sms[:, :, _], torch.transpose(sms[:, :, _], 0, 1)) for _ in range(self.n_classes)], dim=0)
            orth_loss = torch.mean((cps - diag_tensor)**2)
            
            loss = class_loss + specific_loss + orth_loss 
            
            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"specific": specific_loss.item(), "class": class_loss.item() 
                             },
                            {"class": torch.sum(cls_pred == class_l.data).item()
                            },
                            data.shape[0])
            del loss, class_loss, specific_loss, specific_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                specific_correct, class_correct = self.do_test(loader)
                specific_acc = float(specific_correct) / total
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"specific": specific_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        specific_correct = 0
        class_correct = 0
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            dummy_ids = one_hot(np.zeros(len(data), dtype=np.int32), 3)
            specific_logit, class_logit = self.model(data, torch.tensor(dummy_ids, dtype=torch.float, device='cuda'))
            _, cls_pred = class_logit.max(dim=1)
            _, specific_pred = specific_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            specific_correct += torch.sum(specific_pred == class_l.data)
        print (self.model.embs, self.model.cs_wt)
        return specific_correct, class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        k = 512
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
