# 用测试机测模型，并将输出结果保存至以下路径
import os
path_pred_result = os.path.join("output","pred_result")
if not os.path.isdir(path_pred_result):
    os.makedirs(path_pred_result)
import torch
import yaml
from torch.utils.data import DataLoader
from utils import LoadData, WriteData
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
from tqdm import tqdm
import pandas as pd


def test(dataloader, model, device):
    pred_list = []
    # 将模型转为验证模式，只需要前向传播
    model.eval()
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in tqdm(dataloader):
            # 将数据转到GPU
            X, y = X.to(device), y.to(device)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            pred_softmax = torch.softmax(pred, 1).cpu().numpy()
            pred_list.append(pred_softmax.tolist()[0])
        return pred_list


if __name__ == '__main__':

    pathConfigYaml = os.path.join("information", "config.yaml")
    with open(pathConfigYaml, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    myNet = config["myNet"]
    dataset_label = config["dataset_label"]
    classNum = len(dataset_label)

    batch_size = 1

    # 给测试集创建一个数据集加载器
    pathTestset = os.path.join("datasetGroup", "testGroup.txt")
    print(f"Now test the model by Images: testGroup.txt")
    testData = LoadData(pathTestset, False)
    test_dataloader = DataLoader(dataset=testData, num_workers=4, pin_memory=True, batch_size=batch_size)

    # 如果显卡可用，则用显卡进行测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
    if myNet == "resnet18":
        model = resnet18(num_classes=classNum)
    elif myNet == "resnet34":
        model = resnet34(num_classes=classNum)
    elif myNet == "resnet50":
        model = resnet50(num_classes=classNum)
    elif myNet == "resnet101":
        model = resnet101(num_classes=classNum)
    elif myNet == "resnet152":
        model = resnet152(num_classes=classNum)
    elif myNet == "mobilenet_v2":
        model = mobilenet_v2(num_classes=classNum)
    else:
        model = resnet18(num_classes=classNum)
        print(f"不存在模型{myNet},将默认使用resnet18")

    # 自动读取模型名字
    save_root = "output"
    model_name = myNet + "_" + config["mark"]
    model_path = os.path.join(save_root, model_name + "_best.pth")
    # 你也可以手动指定模型
    # model_path = os.path.join(save_root, "resnet18_best.pth")

    print(f"Now you use the model: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # 获取模型输出
    pred_list = test(test_dataloader, model, device)
    print("pred_list = ", pred_list)

    df_pred = pd.DataFrame(data=pred_list, columns=dataset_label)
    print(df_pred)
    df_pred.to_csv(os.path.join(path_pred_result, "pred_result.csv"), encoding='gbk', index=False)
