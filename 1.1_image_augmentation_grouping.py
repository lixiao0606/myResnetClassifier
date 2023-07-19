# 数据增强 和 分组

import os
import yaml
import random
import cv2
import numpy as np
from tqdm import tqdm

with open(os.path.join("information", 'config.yaml'), 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

dataset_label = config["dataset_label"]
val_k = config["val_k"]
classNum = len(dataset_label)

# path
pathProject = ".\\"
pathDataset = os.path.join(pathProject, "dataset")
pathEnhanceRoot = os.path.join(pathProject, "datasetEnhance")
if os.path.isdir(pathEnhanceRoot) == False:
    os.makedirs(pathEnhanceRoot)

for index, value in enumerate(dataset_label):
    tempPathEnhance = os.path.join(pathEnhanceRoot, value)
    if os.path.isdir(tempPathEnhance) == False:
        os.makedirs(tempPathEnhance)

pathDatasetGroupRoot = os.path.join(pathProject, r"datasetGroup")
if os.path.isdir(pathDatasetGroupRoot) == False:
    os.makedirs(pathDatasetGroupRoot)
prefixCondition_TrainTxt_ = os.path.join(pathDatasetGroupRoot, "trainGroup")
prefixCondition_ValTxt_ = os.path.join(pathDatasetGroupRoot, "valGroup")


# 图像增强方式
# 图片亮度改变：gamma矫正
def transGamma(img, gamma):  # gamma大于1时图片变暗，小于1图片变亮
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)

# 图片亮度改变：线性
def transLinear(img, power):
    # power大于1时图片变亮，小于1时图片变暗
    img = np.float32(img) * power // 1
    img[img > 255] = 255
    img = np.uint8(img)
    return img

# 增加噪声
def transAddGaussianNoise(img, lever):
    # 获取图像尺寸
    height, width = img.shape[:2]
    # 定义噪声参数
    mean = 0
    var = 50  # 均值0.1*random.random()
    sigma = var ** (0.4 + lever * 0.1)  # 方差var ** 0.5
    # 生成高斯噪声
    gauss = np.random.normal(mean, sigma, (height, width, 3))
    # 将图像和噪声相加
    noisy_img = np.clip(img[:, :, :3] + gauss, 0, 255)
    return noisy_img

# 水平翻转
def transHorizontal(image):
    return cv2.flip(image, 1, dst=None)  # 水平镜像

# 垂直翻转
def transVertical(image):
    return cv2.flip(image, 0, dst=None)  # 垂直镜像

# 随机裁剪
def transRamdonCrop(image):
    h = len(image)
    w = len(image[0])
    h1 = random.choice(range(0, h // 2))
    w1 = random.choice(range(0, w // 2))
    image = image[h1:h // 2 + h1, w1:w // 2 + w1]
    image = cv2.resize(image, (w, h))
    return image

# 数据增强与数据集制作:图片路径,类别标签，训练集图片路径保存的列表，验证集……(pathImg.jpg,3,[],[])
def ImageEnhance(imagePath, labelId, trainVector, valVector):
    imgLabel = dataset_label[labelId]
    # print("图像增强："+ image)
    # 这样读取和保存可以防止中文报错
    imgRead = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)  # 读取图片
    imgName = imagePath.split("\\")[-1].split(".")[-2]

    # 保存原图
    imgTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_Original.jpg")
    cv2.imencode('.jpg', imgRead)[1].tofile(imgTempPath)  # 保存图片
    # valVector.append(imgTempPath + '\t' + str(labelId) + '\n')
    valVector.append(imagePath + '\t' + str(labelId) + '\n') # 验证集和测试集 直接使用原数据集图片
    trainVector.append(imgTempPath + '\t' + str(labelId) + '\n')

    # 执行随机Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
    if labelId in []:
        imgCorrected = transGamma(imgRead, 0.7 + random.random() * 0.1)
        imgCorrectedTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_Correct_ram0.7.jpg")
        cv2.imencode('.jpg', imgCorrected)[1].tofile(imgCorrectedTempPath)
        trainVector.append(imgCorrectedTempPath + '\t' + str(labelId) + '\n')

        imgCorrected = transGamma(imgRead, 1.7 + random.random() * 0.2)
        imgCorrectedTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_Correct_ram1.7.jpg")
        cv2.imencode('.jpg', imgCorrected)[1].tofile(imgCorrectedTempPath)
        trainVector.append(imgCorrectedTempPath + '\t' + str(labelId) + '\n')

    # 执行固定Gamma矫正
    if labelId in []:
        imgCorrected = transGamma(imgRead, 0.45)
        imgCorrectedTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_Correct_0.45.jpg")
        cv2.imencode('.jpg', imgCorrected)[1].tofile(imgCorrectedTempPath)
        trainVector.append(imgCorrectedTempPath + '\t' + str(labelId) + '\n')

        imgCorrected = transGamma(imgRead, 2.2)
        imgCorrectedTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_Correct_2.2.jpg")
        cv2.imencode('.jpg', imgCorrected)[1].tofile(imgCorrectedTempPath)
        trainVector.append(imgCorrectedTempPath + '\t' + str(labelId) + '\n')

    if labelId in []:
        imgGaussianNoise = transAddGaussianNoise(imgRead, 1)
        imgGaussianNoiseTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_ImgGaussianNoise1.jpg")
        cv2.imencode('.jpg', imgGaussianNoise)[1].tofile(imgGaussianNoiseTempPath)
        trainVector.append(imgGaussianNoiseTempPath + '\t' + str(labelId) + '\n')

    if labelId in []:
        imgGaussianNoise = transAddGaussianNoise(imgRead, 2)
        imgGaussianNoiseTempPath = os.path.join(pathEnhanceRoot, imgLabel, imgName + "_ImgGaussianNoise2.jpg")
        cv2.imencode('.jpg', imgGaussianNoise)[1].tofile(imgGaussianNoiseTempPath)
        trainVector.append(imgGaussianNoiseTempPath + '\t' + str(labelId) + '\n')


if __name__ == '__main__':

    # 生成文件夹datasetEnhance装所有的增强过后的图片
    # 生成txt文件 共5+5+1个：
    #   5个txt装训练集的图片的位置（增强后的）
    #       trainGroup0.txt
    #   5个txt装验证集的图片的位置
    #       val_data_group0.txt
    #   1个txt装测试集的图片的位置
    #       testGroup.txt

    # 原数据集datasetList,双重列表：6个类，以及每个类下所有文件的路径
    # 路径加上文件夹前缀，分类别进行打乱
    datasetList = [os.listdir(os.path.join(pathDataset, dataset_label[i])) for i in range(len(dataset_label))]
    for i in range(len(dataset_label)):
        tempPath = os.path.join(pathDataset, dataset_label[i])
        for j, eachImgPath in enumerate(datasetList[i]):
            datasetList[i][j] = os.path.join(tempPath, eachImgPath)
        random.shuffle(datasetList[i])

    # print(datasetList)
    # 此时datasetList分类别保存：所有图片的正确路径（打乱）

    # 分组信息保存在列表
    # 目前Train的第0组保存第0组的增强数据，而应该保存1234组增强数据作为训练，未增强的0作为验证,后续进行调整
    list_TrainGroup_ori = [[] for i in range(val_k + 1)]
    list_TrainGroup = [[] for i in range(val_k + 1)]
    list_ValGroup = [[] for i in range(val_k + 1)]
    # 数据增强与数据集制作
    print("图像增强中：")
    for i in range(len(dataset_label)):
        for j, eachImgPath in tqdm(enumerate(datasetList[i])):
            group = j % (val_k + 1)
            ImageEnhance(eachImgPath, i, list_TrainGroup_ori[group], list_ValGroup[group])

    # 目前Train的第0组保存第0组的增强数据，而应该保存1234组增强数据作为训练，未增强的0作为验证,进行调整
    for group in range(val_k):
        for group2 in range(val_k):
            if not group == group2:
                list_TrainGroup[group].extend(list_TrainGroup_ori[group2])

    # 打乱训练集
    for group in range(val_k):
        random.shuffle(list_TrainGroup[group])
    # 训练集验证集写入本地txt
    for group in range(val_k):
        temp_train_txt = prefixCondition_TrainTxt_ + str(group) + ".txt"
        temp_val_txt = prefixCondition_ValTxt_ + str(group) + ".txt"
        with open(temp_train_txt, 'w', encoding='UTF-8') as f:
            f.writelines(list_TrainGroup[group])
        with open(temp_val_txt, 'w', encoding='UTF-8') as f:
            f.writelines(list_ValGroup[group])
    # 测试集写入本地txt
    with open(os.path.join(pathDatasetGroupRoot, "testGroup.txt"), 'w', encoding='UTF-8') as f:
        f.writelines(list_ValGroup[val_k])

    # 对于每个类别，分别做了以下处理
    # 1.随机打乱  random.shuffle(ori_data[0])
    # 2.对于 ori_data[0]中的每一个数据，它所在的组为其下标%5
    # 3.对于 ori_data[0]中的每一个数据，将其转写到对应组的val_data_group0.txt文件中（包含路径和标签）
    # 4.对于 ori_data[0]中的每一个数据，将其进行数据增强，增强的同时，把增强后的[路径,标签]，转写到一个新的列表train_data = [5个组]中，之后记得再打乱
