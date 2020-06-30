import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn

from models.alexnet import Id
from models.model_utils import ReverseLayerF


class AlexNetCaffe(nn.Module):
    def __init__(self, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        classes = n_classes
        K = 2
        
        self.sms = torch.nn.Parameter(torch.normal(0., 1e-3, size=[K, 4096, classes], dtype=torch.float, device='cuda'), requires_grad=True)
        self.sm_biases = torch.nn.Parameter(torch.normal(0., 1e-3, size=[K, classes], dtype=torch.float, device='cuda'), requires_grad=True)
    
        self.embs = torch.nn.Parameter(torch.normal(mean=0., std=1e-1, size=[3, K-1], dtype=torch.float, device='cuda'), requires_grad=True)
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=0., std=1e-4, size=[], dtype=torch.float, device='cuda'), requires_grad=True)
        
    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.jigsaw_classifier.parameters()
                                 , self.class_classifier.parameters()#, self.domain_classifier.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, uids, lambda_val=0):
        x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        # 8th Layer: FC and return unscaled activations
        logits_common = torch.matmul(x, w_c) + b_c

        c_wts = torch.matmul(uids, self.embs)
        # B x K
        batch_size = uids.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float, device='cuda')*self.cs_wt, c_wts), 1)
        w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d
        
        return logits_specialized, logits_common

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def caffenet(classes):
    model = AlexNetCaffe(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    print ("Loaded from: %s" % os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    print ("Test: ", state_dict["classifier.fc7.weight"])
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model
