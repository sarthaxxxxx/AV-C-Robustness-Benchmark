import torch
import torchvision

from .audio_net import  ANet
from .vision_net import  Resnet
from .cls_net import Classifier_Concat
from .criterion import BCELoss, CELoss


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, weights=''):
        net_sound = ANet()
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))
        return net_sound

    def build_frame(self, pool_type='avgpool', weights=''):
        pretrained=True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet(original_resnet, pool_type=pool_type)
        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_classifier(self, cls_num, weights=''):
        net = Classifier_Concat(cls_num)
        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_grounding')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'ce':
            net = CELoss()
        else:
            raise Exception('Architecture undefined!')
        return net
