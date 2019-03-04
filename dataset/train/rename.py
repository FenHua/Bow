import os
import scipy.misc
image_dir_path="/home/yhq/PycharmProjects/BOW/dataset/train"
neg_images_path="/home/yhq/PycharmProjects/BOW/dataset/dog"
i=0
for image_name in os.listdir(image_dir_path):
        image = scipy.misc.imread(os.path.join(image_dir_path, image_name))  # 读图
        scipy.misc.imsave(os.path.join(neg_images_path, 'dog.'+str(i) + '.' + 'jpg'), image)
        i=i+1

