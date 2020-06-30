from models import caffenet
from models import caffenet_csd
from models import mnist
from models import patch_based
from models import alexnet
from models import resnet
from models import resnet_csd

nets_map = {
    'caffenet': caffenet.caffenet,
    'caffenet_csd': caffenet_csd.caffenet,
    'alexnet': alexnet.alexnet,
    'resnet18': resnet.resnet18,
    'resnet18_csd': resnet_csd.resnet18,
    'resnet50': resnet.resnet50,
    'lenet': mnist.lenet
}


def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
