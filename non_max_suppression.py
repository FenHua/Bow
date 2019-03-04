# -*- coding: utf-8 -*-

import numpy as np
'''
1. 按打分最高到最低将BBox排序，例如：A B C D E F
2. A的分数最高，保留，从B-E与A分别求重叠率IoU，假设B、D与A的IoU大于阈值，那么B和D可以认为是重复标记去除
3. 余下C E F，重复前面两步
'''


def nms(boxes, threshold):
    # boxes：边界框，数据为list类型，形状为[x1,y1,x2,y2,score]
    # threshold：IOU阈值
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    # 计算边界框区域大小，并按照score进行倒叙排序
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    keep = []  # keep为最后保留的边框
    while len(idxs) > 0:
        # idxs[0]是当前分数最大的窗口，肯定保留
        i = idxs[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[idxs[1:]] - inter)  # 计算iou值
        inds = np.where(ovr <= threshold)[0]  # 保留所有与窗口i的iou值小于threshold值的窗口的index
        idxs = idxs[inds + 1]
        # inds里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比inds长度少1(不包含i)，所以inds+1对应到保留的窗口
    return boxes[keep]