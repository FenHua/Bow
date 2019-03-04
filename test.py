# -*- coding: utf-8 -*-
# 使用BOW+SVM进行滑动窗口目标检测
import cv2
import numpy as np
from detector import BOW
from pyramid import pyramid, silding_window
from non_max_suppression import nms
import os


def prepare_data(rootpath, samples):
    '''
    加载数据集
    args：
        rootpath：数据集所在的根目录
                要求在该路径下，存放数据，每一类使用一个文件夹存放，文件名即为类名
        sample：指定获取的每一类样本长度

    return：
        train_path：训练集路径 list类型  [['calss0-1','calss0-2','calss0-3','class0-4',...]
                                         ['calss1-1','calss1-2','calss1-3','class1-4',...]
                                         ['calss2-1','calss2-2','calss2-3','class2-4',...]
                                         ['calss3-1','calss3-2','calss3-3','class3-4',...]
        labels：每一个样本类别标签 list类型 [[0,0,0,0]...
                                            [1,1,1,1]...
                                            [2,2,2,2]...
                                            [3,3,3,3]...
                                            ...]
        classes：每一个类别对应的名字 list类型
    '''
    files = []
    labels = []
    classes = [x for x in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, x))] # 获取rootpath下的所有文件夹
    for idx in range(len(classes)):
        # 遍历每个类别样本
        path = os.path.join(rootpath, classes[idx])  # 获取当前类别文件所在文件夹的全路径
        filelist = [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))] # 遍历每一个文件路径
        files.append(filelist[:samples])  # 追加到字典
        labels.append([idx] * samples)
    return files, labels, classes


if __name__ == '__main__':
    # 训练或者直接加载训练好的模型
    is_training = False
    bow = BOW()
    if is_training:
        samples = 450  # 样本个数 越大，训练准确率相对越高
        rootpath = './dataset/plane'  # 根路径
        files, labels, classes = prepare_data(rootpath, samples)  # 获取训练集图片路径
        bow.fit(files, labels, 100, 200) # k越大，训练准确率相对越高
        bow.save('./svm.mat') # 保存模型
    else:
        bow.load('./svm.mat') # 加载模型

    # 测试样本数量，测试结果
    start_index = 100
    test_samples = 200
    test_results = []
    # 指定测试图像路径
    rootpath = './dataset/plane' # 根路径
    files, labels, classes = prepare_data(rootpath, 400) # 训练集图片路径
    for j in range(len(files)):
        for i in range(start_index, start_index + test_samples):
            # 预测
            img = cv2.imread(files[j][i])
            label, score = bow.predict(img)
            if label is None:
                continue
            # print(files[j][i],label,labels[j][i])
            if label == labels[j][i]:
                test_results.append(True)
            else:
                test_results.append(False)
    test_results = np.asarray(test_results, dtype=np.float32)
    accuracy = np.mean(test_results) # 计算准确率
    print('测试准确率为：', accuracy)

    # 利用滑动窗口进行目标检测
    w, h = 40, 40  # 滑动窗口大小
    test_img = './dataset/plane/1.jpg'
    img = cv2.imread(test_img)
    rectangles = []
    counter = 1
    scale_factor = 1.2
    font = cv2.FONT_HERSHEY_PLAIN
    # 图像金字塔
    for resized in pyramid(img.copy(), scale_factor, (img.shape[1] // 2, img.shape[1] // 2)):
        # print(resized.shape)
        scale = float(img.shape[1]) / float(resized.shape[1])  # 图像缩小倍数
        # 遍历每一个滑动区域
        for (x, y, roi) in silding_window(resized, 10, (w, h)):
            if roi.shape[1] != w or roi.shape[0] != h:
                continue
            try:
                label, score = bow.predict(roi)
                if label == 1:
                    if score < -1:
                        # print(label,score)
                        # 获取相应边界框的原始大小
                        rx, ry, rx2, ry2 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
                        rectangles.append([rx, ry, rx2, ry2, -1.0 * score])
            except:
                pass
            counter += 1
    windows = np.array(rectangles)
    boxes = nms(windows, 0.05)
    for x, y, x2, y2, score in boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.putText(img, '%f' % score, (int(x), int(y)), font, 1, (0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()