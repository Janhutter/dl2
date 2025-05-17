import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import  ResNet, Bottleneck, BasicBlock
from .wideresnet import WideResNet
from .vit import VisionTransformer

def build_model_res50gn(group_norm, num_classes):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(group_norm, planes)
    net = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes, norm_layer=gn_helper)
    return net

def build_model_res18bn(num_classes):
    print('Building model...')
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=nn.BatchNorm2d)

def build_model_wrn2810bn(num_classes):
    print('Building model...')
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropRate=0.0)

def build_vit(num_classes, model_name="google/vit-base-patch16-224-in21k", dropout_rate=0.0):
    model = VisionTransformer(
        num_classes=num_classes,
        model_name=model_name,
        dropout_rate=dropout_rate
    )
    return model

    
# def build_model_wrn2810bn_TET(num_classes):
#     print('Building model...')
#     return WideResNet_TET(depth=28, widen_factor=10, num_classes=num_classes, dropRate=0.0)