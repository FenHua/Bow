# -*- coding: utf-8 -*-
'''
词袋模型BOW+SVM 目标识别
以狗和猫数据集二分类为例
如果是狗 返回True
如果是猫 返回False
'''
import numpy as np
import cv2

class BOW(object):

    def __init__(self, ):
        self.feature_detector = cv2.xfeatures2d.SIFT_create()  # 创建用于获取特征的sift对象
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()  # 用于关键点描述符提取

    def path(self, cls, i):
        return '%s/%s/%s.%d.jpg' % (self.train_path, cls, cls, i + 1)  # 用于获取图片的完整路径

    def fit(self, train_path, k):
        '''
        训练函数
        输入：
            train_path：单张图片路经
            k：类簇的个数
        '''
        self.train_path = train_path
        # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
        pos = 'dog'
        neg = 'cat'
        length = 10  # 指定用于提取词汇字典的样本数
        for i in range(length):
            # 合并特征数据，从每类数据中获取length的数据，通过聚类创建视觉词汇
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(pos, i)))
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(neg, i)))
        voc = bow_kmeans_trainer.cluster() # 进行k-means聚类，返回词汇字典 也就是聚类中心
        # print(type(voc), voc.shape) # 输出词汇字典  <class 'numpy.ndarray'> (40, 128)，其中k=40

        # FLANN快速最近邻搜索包  参数algorithm用来指定匹配所使用的算法，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1, tree=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})

        # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
        self.bow_img_descriptor_extractor.setVocabulary(voc)
        traindata, trainlabels = [], []  # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        for i in range(400):
            # 这里取800张图像做训练
            traindata.extend(self.bow_descriptor_extractor(self.path(pos, i)))
            trainlabels.append(1)  # 按照下面的方法生成相应的正负样本图片的标签 1：正匹配  -1：负匹配
            traindata.extend(self.bow_descriptor_extractor(self.path(neg, i)))
            trainlabels.append(-1)
        self.svm = cv2.ml.SVM_create()  # 创建一个SVM对象
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setGamma(0.5)
        self.svm.setC(30)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))  # 以行计数进行训练

    def predict(self, img_path):
        data = self.bow_descriptor_extractor(img_path) # 提取图片的BOW特征描述
        res = self.svm.predict(data)
        print(img_path, '\t', res[1][0][0])
        if res[1][0][0] == 1.0:
            return True  # 如果是狗 返回True
        else:
            return False  # 如果是猫，返回False

    def sift_descriptor_extractor(self, img_path):
        # 特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        im = cv2.imread(img_path, 0)  # flag=0，8位深度，1通道
        return self.descriptor_extractor.compute(im, self.feature_detector.detect(im))[1]

    def bow_descriptor_extractor(self, img_path):
        # 提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        im = cv2.imread(img_path, 0)
        return self.bow_img_descriptor_extractor.compute(im, self.feature_detector.detect(im))


if __name__ == '__main__':
    test_samples = 100  # 测试样本数量，测试结果
    test_results = np.zeros(test_samples, dtype=np.bool)
    train_path = './dataset/train' # 训练集图片路径  狗和猫两类  进行训练
    bow = BOW()
    bow.fit(train_path, 40)
    for index in range(test_samples):
        dog = './dataset/train/dog/dog.{0}.jpg'.format(index)  # 指定测试图像路径
        dog_img = cv2.imread(dog)
        dog_predict = bow.predict(dog)
        test_results[index] = dog_predict
    accuracy = np.mean(test_results.astype(dtype=np.float32))  # 计算准确率
    print('测试准确率为：', accuracy)
    # 可视化最后一个
    font = cv2.FONT_HERSHEY_SIMPLEX
    if test_results[0]:
        cv2.putText(dog_img, 'Dog Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('dog_img', dog_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
