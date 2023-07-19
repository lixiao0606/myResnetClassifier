"""
    采用迁移学习
    k 折交叉验证
    验证不进入训练过程

"""
import tqdm
import random
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData, WriteData
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2  # ResNet系列
import yaml
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
import pandas as pd  # pip install pandas
import os

# path
pathProject = ".\\"
pathDatasetGroup = os.path.join(pathProject, r"datasetGroup")
# 前缀
prefix_TrainTxt_ = os.path.join(pathDatasetGroup, "trainGroup")
prefix_ValTxt_ = os.path.join(pathDatasetGroup, "valGroup")


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    avg_loss = 0
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):  # 固定格式：batch：第几批数据，不是批次大小，（X，y）：数值用括号

        # print(size)
        # 将数据存到显卡
        X, y = X.to(device), y.to(device)
        # 得到预测的结果pred
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss
        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每训练10次，输出一次当前信息
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print("\r" + f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="", flush=True)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # 当一个epoch完了后返回平均 loss
    avg_loss /= size
    avg_loss = avg_loss.detach().cpu().numpy()
    return avg_loss


def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）

        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.to(device), y.to(device)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数(针对分类)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss



if __name__ == '__main__':

    pathConfigYaml = os.path.join("information", "config.yaml")
    with open(pathConfigYaml, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    batch_size = config["batch_size"]
    val_k = config["val_k"]
    myNet = config["myNet"]
    dataset_label = config["dataset_label"]
    classNum = len(dataset_label)
    epochs = config["epochs"]
    loss_ = config["loss_"]

    save_root = "output"
    model_name = myNet + "_" + config["mark"]

    back_up_root = os.path.join(save_root, "modelBackUp")
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    if not os.path.isdir(back_up_root):
        os.makedirs(back_up_root)

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 给训练集和测试集分别创建一个数据集加载器
    train_data = []
    valid_data = []
    train_dataloader = []
    valid_dataloader = []

    for group in range(val_k):
        train_txt_prefix = prefix_TrainTxt_
        val_txt_prefix = prefix_ValTxt_

        temp_train_txt = train_txt_prefix + str(group) + ".txt"
        temp_val_txt = val_txt_prefix + str(group) + ".txt"
        train_data.append(LoadData(temp_train_txt, True))
        valid_data.append(LoadData(temp_val_txt, False))
        train_dataloader.append(
            DataLoader(dataset=train_data[group], num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True))
        valid_dataloader.append(
            DataLoader(dataset=valid_data[group], num_workers=4, pin_memory=True, batch_size=batch_size))

    # resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
    if myNet == "resnet18":
        finetune_net = resnet18(pretrained=True)
    elif myNet == "resnet34":
        finetune_net = resnet34(pretrained=True)
    elif myNet == "resnet50":
        finetune_net = resnet50(pretrained=True)
    elif myNet == "resnet101":
        finetune_net = resnet101(pretrained=True)
    elif myNet == "resnet152":
        finetune_net = resnet152(pretrained=True)
    elif myNet == "mobilenet_v2":
        finetune_net = mobilenet_v2(pretrained=True)
    else:
        finetune_net = resnet18(pretrained=True)
        print(f"不存在模型{myNet},将默认使用resnet18")


    # nn.Linear就是神经网络中的线性层，可以实现形如y=X*weight^T+b的功能
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features,
                                classNum)

    # Xavier初始化，解决随机初始化的问题，尽可能让输入和输出服从相同分布，这样就能够避免后面层的激活函数的输出值趋向于0。
    nn.init.xavier_normal_(finetune_net.fc.weight)

    parms_1x = [value for name, value in finetune_net.named_parameters()
                if name not in ["fc.weight", "fc.bias"]]

    # 最后一层fc调整为10倍学习率
    parms_10x = [value for name, value in finetune_net.named_parameters()
                 if name in ["fc.weight", "fc.bias"]]

    finetune_net = finetune_net.to(device)
    # 定义损失函数为交叉熵
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    learning_rate = 1e-4
    optimizer = torch.optim.Adam([
        {
            'params': parms_1x
        },
        {
            'params': parms_10x,
            'lr': learning_rate * 10
        }], lr=learning_rate)

    # 开始训练
    for t in range(epochs):
        temp_group = random.choice(range(val_k))
        print(f"Epoch {t + 1}\n-------------------------------")
        time_start = time.time()
        # 训练该epoch
        avg_loss = train(train_dataloader[temp_group], finetune_net, loss_fn, optimizer, device)
        time_end = time.time()
        print(f"train time: {(time_end - time_start)}")
        # 验证该epoch
        val_accuracy, val_loss = validate(valid_dataloader[temp_group], finetune_net, loss_fn, device)

        # 写入数据
        WriteData(os.path.join(save_root, model_name + ".txt"),
                  "epoch", t,
                  "train_loss", avg_loss,
                  "val_loss", val_loss,
                  "val_accuracy", val_accuracy)

        # 保存模型参数
        torch.save(finetune_net.state_dict(),
                   os.path.join(back_up_root, model_name + "_epoch" + str(t) + ".pth"))
        # torch.save(finetune_net.state_dict(), save_root + model_name + "_last.pth")
        if avg_loss < loss_:
            loss_ = avg_loss
            torch.save(finetune_net.state_dict(), os.path.join(save_root, model_name + "_best.pth"))
