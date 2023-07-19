# 计算均值和方差

from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms as T
# 进度条
from tqdm import tqdm
import yaml
import os
import numpy as np

# 压缩图片尺寸为短边224，为了运算速度
transform = T.Compose([
    # T.RandomResizedCrop(224),
    T.ToTensor(),
])


def getStat(train_data):
    # 定义一个数据加载器，传入训练数据，bs1，不洗牌，不多线程
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    mean = torch.zeros(3)  # 均值
    std = torch.zeros(3)  # 方差
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()  # N, C, H ,W：C为3个通道
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


def VectorToYaml(vector, pathYamlFile):
    # 将列表保存进meanVariance.yaml文件
    with open(pathYamlFile, 'w') as file:
        yaml.dump(vector, file)
    # 从JSON文件中读取列表
    with open(pathYamlFile, 'r', encoding='utf-8') as file:
        my_list = yaml.safe_load(file)
    # 打印读取的列表
    print(my_list)


if __name__ == '__main__':
    pathEnhance = "datasetEnhance"

    trainDataset_Enhance = ImageFolder(root=pathEnhance, transform=transform)
    meanVariance = list(getStat(trainDataset_Enhance))
    # 先将list转换成numpy.array，再将numpy.array转换成list(解决写入问题)
    meanVariance = np.array(meanVariance).tolist()

    # 数据保存至yaml文件
    pathInformation = "information"
    if not os.path.isdir(pathInformation):
        os.makedirs(pathInformation)
    yamlPath = os.path.join(pathInformation, "meanVariance.yaml")
    tempData = {
        "mean": meanVariance[0],
        "variance": meanVariance[1]
    }
    VectorToYaml(tempData, yamlPath)
