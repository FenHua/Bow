# -*- coding: utf-8 -*-
# 词袋模型BOW+SVM 目标检测
import numpy as np
import cv2
import pickle
import os


class BOW(object):

    def __init__(self, ):
        self.feature_detector = cv2.xfeatures2d.SIFT_create()  # 创建一个SIFT对象  用于关键点提取
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()  # 创建一个SIFT对象  用于关键点描述符提取

    def fit(self, files, labels, k, length=None):
        '''
        训练函数
        函数输入参数：files：训练集图片路径 list类型; labes：对应的每个样本的标签;
            k：k-means参数k; 参数length：表示用于得到词典所采用的样本数
        '''
        classes = len(files)  # 类别数
        samples = len(files[0])  # 样本数量
        if length is None:
            length = samples
        elif length > samples:
            length = samples
        # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1, tree=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)  # 创建BOW训练器，指定k-means参数k
        print('building BOWKMeansTrainer...')
        # 创建视觉词汇，每个类从数据集中读取length张图片
        for j in range(classes):
            for i in range(length):
                descriptor = self.sift_descriptor_extractor(files[j][i])
                if not descriptor is None:
                    # 有一些图像会抛异常,主要是因为该图片没有sift描述符
                    bow_kmeans_trainer.add(descriptor)
        self.voc = bow_kmeans_trainer.cluster()  # 进行聚类，返回类簇中心点(视觉单词)
        print(type(self.voc), self.voc.shape) # 输出词汇字典  <class 'numpy.ndarray'> (40, 128)

        # 初始化bow提取器,用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
        self.bow_img_descriptor_extractor.setVocabulary(self.voc)

        print('adding features to svm trainer...')
        # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        # 按照下面的方法生成相应的正负样本图片的标签
        traindata, trainlabels = [], []
        for j in range(classes):
            for i in range(samples):
                descriptor = self.bow_descriptor_extractor(files[j][i])
                if not descriptor is None:
                    traindata.extend(descriptor)
                    trainlabels.append(labels[j][i])
        self.svm = cv2.ml.SVM_create() # 创建一个SVM对象
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setGamma(0.5)
        self.svm.setC(30)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels)) # 使用训练数据和标签进行训练

    def save(self, path):
        # 将模型保存到指定路径
        print('saving  model....')
        self.svm.save(path)  # 保存svm模型
        f1 = os.path.join(os.path.dirname(path), 'dict.pkl')  # 保存bow模型
        with open(f1, 'wb') as f:
            pickle.dump(self.voc, f)

    def load(self, path):
        # 加载模型
        print('loading  model....')
        self.svm = cv2.ml.SVM_load(path)  # 加载svm模型
        f1 = os.path.join(os.path.dirname(path), 'dict.pkl')  # 加载bow模型
        with open(f1, 'rb') as f:
            voc = pickle.load(f)
            # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
            flann_params = dict(algorithm=1, tree=5)
            flann = cv2.FlannBasedMatcher(flann_params, {})
            # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
            self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
            self.bow_img_descriptor_extractor.setVocabulary(voc)

    def predict(self, img):
        # 样本预测，函数返回label和score：置信度 分数越低，置信度越高，表示属于该类的概率越大
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰色
        # 提取图片的BOW特征描述
        # data = self.bow_descriptor_extractor(img_path)
        keypoints = self.feature_detector.detect(img)
        if keypoints:
            data = self.bow_img_descriptor_extractor.compute(img, keypoints)
            _, result = self.svm.predict(data)
            label = result[0][0] # 所属标签
            # 设置标志位 获取预测的评分  分数越低，置信度越高，表示属于该类的概率越大
            a, res = self.svm.predict(data, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            score = res[0][0]
            # print('Label:{0}  Score：{1}'.format(label,score))
            return label, score
        else:
            return None, None

    def sift_descriptor_extractor(self, img_path):
        # 提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；img_path：图像全路径
        im = cv2.imread(img_path, 0)
        keypoints = self.feature_detector.detect(im)
        if keypoints:
            return self.descriptor_extractor.compute(im, keypoints)[1]
        else:
            return None

    def bow_descriptor_extractor(self, img_path):
        # 提取图像的BOW特征描述(即利用视觉词袋量化图像特征)，img_path：图像全路径
        im = cv2.imread(img_path, 0)
        keypoints = self.feature_detector.detect(im)
        if keypoints:
            return self.bow_img_descriptor_extractor.compute(im, keypoints)
        else:
            return None