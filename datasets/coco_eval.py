# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_type='bbox'):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_type = iou_type

        self.img_ids = []
        self.outputs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))       # 计算IOU时, 需要通过Img_id来匹配标注文件中该图像的GT
        self.img_ids.extend(img_ids)
        results = self.prepare_for_coco_detection(predictions)    # 整理模型的输出结果, 将输出结果中每个box的信息, 聚合起来, 一个box的输出信息都放到一个dict中
        self.outputs.extend(results)                              # self.outputs是一个list, 里面每一个元素都是一个dict结构, 包含了一个bbox的信息

    def getResult(self):
        coco_dt = COCO.loadRes(self.coco_gt, self.outputs)        # 完善 prepare_for_coco_detection 构建的简单dict 中的信息, 计算更多信息, 如area, iscrowd, id
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()   

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()    # 训练模型时传入模型的box坐标是(左上, 右下), 因此模型输出也是(左上, 右下), 为了借助pycocotools计算mAP, 还需要将坐标转换为 (左上, 宽高) 的形式
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)