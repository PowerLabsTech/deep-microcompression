import torch
from torch import nn

from ..models.utils import convert_from_sequential_torch_to_dmc




def get_model():

    pretraind_vqq_13_torch = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg13_bn", pretrained=True)
    vgg_13_torch_model = pretraind_vqq_13_torch.features + nn.Sequential(nn.Flatten()) + pretraind_vqq_13_torch.classifier
    vgg_13_dmc_model = convert_from_sequential_torch_to_dmc(vgg_13_torch_model)
    return vgg_13_dmc_model