import os

import yaml

save_root = "output"
path_allEpochs_modelsTest = os.path.join(save_root,"allEpochs_modelsTest")
path_all_models_results = os.path.join(path_allEpochs_modelsTest, "all_models_results.txt")
path_all_models_results_yaml = os.path.join(path_allEpochs_modelsTest, "all_models_results.yaml")

import torch
from torch.utils.data import DataLoader
from utils import LoadData, WriteData
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
from tqdm import tqdm
from matplotlib import pyplot as plt

def Draw_Test_Result(Accuracy = False,Precision = False,Recall = False,F1Score = False, epochs = 0,savePath = "output\\unName.png"):
    savefigName = "Test"

    plt.style.use('dark_background')
    plt.title("Model of Each Epoch")
    plt.xlabel("epoch")
    plt.ylabel("Value(%)")


    epoch_list = [i+1 for i in range(epochs)]
    legend_list = []
    legent_name_list = []

    if not Accuracy == False:
        p1 = plt.plot(epoch_list, Accuracy, linewidth=3)
        legend_list.append(p1)
        legent_name_list.append("Accuracy")
        savefigName += "_Acc"

    if not Precision == False:
        p2 = plt.plot(epoch_list, Precision, linewidth=3)
        legend_list.append(p2)
        legent_name_list.append("Precision")
        savefigName += "_P"

    if not Recall == False:
        p3 = plt.plot(epoch_list, Recall, linewidth=3)
        legend_list.append(p3)
        legent_name_list.append("Recall")
        savefigName += "_R"

    if not F1Score == False:
        p4 = plt.plot(epoch_list, F1Score, linewidth=3)
        legend_list.append(p4)
        legent_name_list.append("F1Score")
        savefigName += "_F1"

    plt.legend(legend_list, legent_name_list)

    plt.savefig(os.path.join(savePath,savefigName+".png"))
    plt.show()



if __name__=="__main__":
    pathConfigYaml = os.path.join("information", "config.yaml")
    with open(pathConfigYaml, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    myNet = config["myNet"]
    dataset_label = config["dataset_label"]
    classNum = len(dataset_label)
    epochs = config["epochs"]

    model_name = myNet + "_" + config["mark"]
    # 查看保存了测试结果的yaml文件
    with open(path_all_models_results_yaml, 'r', encoding='utf-8') as file:
        all_models_results = yaml.safe_load(file)
    print(all_models_results)

    accuracy = []
    precision = []
    recall = []
    f1Score = []
    for i in range(epochs):
        accuracy.append(all_models_results[f"model{i}"]["accuracy"]*100)
        precision.append(all_models_results[f"model{i}"]["precision"]*100)
        recall.append(all_models_results[f"model{i}"]["recall"]*100)
        f1Score.append(all_models_results[f"model{i}"]["f1Score"]*100)

    Draw_Test_Result(Accuracy=accuracy,epochs=epochs,savePath = path_allEpochs_modelsTest)
    Draw_Test_Result(Accuracy=accuracy,Precision = precision,Recall = recall,F1Score = f1Score, epochs=epochs, savePath=path_allEpochs_modelsTest)
    Draw_Test_Result(F1Score=f1Score, epochs=epochs, savePath=path_allEpochs_modelsTest)