# 训练时候的中间产物，全部测试一遍，并把测试结果存起来


import os
save_root = "output"
path_allEpochs_modelsTest = os.path.join(save_root,"allEpochs_modelsTest")
if not os.path.isdir(path_allEpochs_modelsTest):
    os.makedirs(path_allEpochs_modelsTest)
path_temp_pred_result = os.path.join(path_allEpochs_modelsTest, "temp_pred_result_by4_1.csv")
path_modelBackUp_folder = os.path.join(save_root, "modelBackUp")
path_all_models_results = os.path.join(path_allEpochs_modelsTest, "all_models_results.txt")
path_all_models_results_yaml = os.path.join(path_allEpochs_modelsTest, "all_models_results.yaml")

import torch
from torch.utils.data import DataLoader
from utils import LoadData, WriteData
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
from tqdm import tqdm

import yaml
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
import pandas as pd  # pip install pandas

pathConfigYaml = os.path.join("information", "config.yaml")
with open(pathConfigYaml, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

myNet = config["myNet"]
dataset_label = config["dataset_label"]
classNum = len(dataset_label)
model_name = myNet + "_" + config["mark"]



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

    batch_size = 1

    # 给测试集创建一个数据集加载器
    pathTestset = os.path.join("datasetGroup", "testGroup.txt")
    print(f"Now test the models by Images: {pathTestset}")
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

    modelList = os.listdir(path_modelBackUp_folder)
    for i in range(len(modelList)):
        tempPath = os.path.join(path_modelBackUp_folder, modelList[i])
        modelList[i] = tempPath

    for i in range(len(modelList)):
        # nowModel = modelList[i]
        nowModel = os.path.join(path_modelBackUp_folder, model_name + "_epoch" + str(i) + ".pth")

        print(f"Now you test the model: {nowModel}")
        model.load_state_dict(torch.load(nowModel))
        model.to(device)

        # 获取模型输出
        pred_list = test(test_dataloader, model, device)
        # print("pred_list = ", pred_list)

        df_pred = pd.DataFrame(data=pred_list, columns=dataset_label)
        # print(df_pred)
        df_pred.to_csv(path_temp_pred_result, encoding='gbk', index=False)

        # 计算
        # ===================

        target_data = pd.read_csv(pathTestset, sep="\t", names=["loc", "type"])
        true_label = [i for i in target_data["type"]]
        predict_data = pd.read_csv(path_temp_pred_result, encoding="GBK")  # ,index_col=0)
        predict_label = predict_data.to_numpy().argmax(axis=1)
        predict_score = predict_data.to_numpy().max(axis=1)

        # 精度，准确率， 预测正确的占所有样本种的比例
        accuracy = accuracy_score(true_label, predict_label)
        print("精度(accuracy): ", accuracy)
        # 查准率P（准确率），precision(查准率)=TP/(TP+FP)
        precision = precision_score(true_label, predict_label, labels=None, pos_label=1,
                                    average='macro')  # 'micro', 'macro', 'weighted'
        print("查准率(precision): ", precision)
        # 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
        recall = recall_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
        print("召回率(recall): ", recall)
        # F1-Score
        f1 = f1_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
        print("F1 Score: ", f1)

        # 写入数据
        WriteData(path_all_models_results,
                  "epoch", i,
                  "accuracy", accuracy,
                  "precision", precision,
                  "recall", recall,
                  "f1Score", f1)
        # # 先将list转换成numpy.array，再将numpy.array转换成list(解决写入问题)
        # meanVariance = np.array(meanVariance).tolist()
        # 将更新后的数据写入 YAML 文件

        temp_accuracy = round(accuracy, 4).tolist()
        temp_precision = round(precision, 4).tolist()
        temp_recall = round(recall, 4).tolist()
        temp_f1 = round(f1, 4).tolist()

        temp_result = {
            f'model{i}': {
                "modelName": nowModel,
                "epoch": i + 1,
                "accuracy": temp_accuracy,
                "precision": temp_precision,
                "recall": temp_recall,
                "f1Score": temp_f1
            }
        }
        with open(path_all_models_results_yaml, 'a') as file:
            yaml.dump(temp_result, file)

        os.remove(path_temp_pred_result)