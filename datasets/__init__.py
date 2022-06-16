# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco

def build_dataset(image_set, args):
    return build_coco(image_set, args)
