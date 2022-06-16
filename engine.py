# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import sys
from tqdm import tqdm
from typing import Iterable

import torch
import util.misc as utils
import torch.distributed as dist
from datasets.coco_eval import CocoEvaluator

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    header = f'Epoch: [{epoch}]'

    tbar = tqdm(data_loader, desc=header, dynamic_ncols=True)
    for samples, targets in tbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        process_loss = reduce_tensor(losses.data)
        tbar.set_postfix(loss=losses.item(), process_total=process_loss.item(), lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, coco_gt, device):
    model.eval()
    criterion.eval()
    header = '[Test] '

    coco_evaluator = CocoEvaluator(coco_gt)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]   # 设置iou的计算阈值

    tbar = tqdm(data_loader, desc=header, dynamic_ncols=True)
    for samples, targets in tbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

    # Cal mAP for predictions 
    coco_evaluator.getResult()

    return coco_evaluator
