# -*- coding: utf-8 -*-
# 图像金字塔

import numpy as np
import cv2

def resize(img, scale_factor):
    # 对图像进行缩放，scale_factor小于1,表示放大操作
    ret = cv2.resize(img, (int(img.shape[1] * (1.0 / scale_factor)), int(img.shape[0] * (1.0 / scale_factor))),
                     interpolation=cv2.INTER_AREA)
    return ret


def pyramid(img, scale=1.5, min_size=(200, 200)):
    '''
    图像金字塔生成器
    输入：
        img：输入图像
        scale：缩放因子
        min_size：图像缩放的最小尺寸 (w,h)
    '''
    yield img
    while True:
        img = resize(img, scale)
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break
        yield img


def silding_window(img, stride, window_size):
    # 滑动窗口函数，返回滑动窗口：x,y,滑动区域图像
    for y in range(0, img.shape[0] - window_size[1], stride):
        for x in range(0, img.shape[1] - window_size[0], stride):
            yield (x, y, img[y:y + window_size[1], x:x + window_size[0]])