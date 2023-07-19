# 3.4: 接着3.3，计算指标和画图
import os
save_root = "output"
path_pred_result = os.path.join(save_root,"pred_result")
path_pred_result_csv = os.path.join(path_pred_result, "pred_result.csv")
import yaml
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
import pandas as pd  # pip install pandas

'''
读取数据
需要读取模型输出的标签（predict_label）以及原本的标签（true_label）
'''

pathConfigYaml = os.path.join("information", "config.yaml")
with open(pathConfigYaml, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

myNet = config["myNet"]
dataset_label = config["dataset_label"]
classNum = len(dataset_label)

# 真实标签所在的文件

pathTestset = os.path.join("datasetGroup", "testGroup.txt")
print(f"Now test the model by Images: testGroup.txt")

target_data = pd.read_csv(pathTestset, sep="\t", names=["loc", "type"])
true_label = [i for i in target_data["type"]]
predict_loc = path_pred_result_csv  # 3.ModelEvaluate.py生成的文件

predict_data = pd.read_csv(predict_loc, encoding="GBK")  # ,index_col=0)

predict_label = predict_data.to_numpy().argmax(axis=1)

predict_score = predict_data.to_numpy().max(axis=1)

'''
    常用指标：精度，查准率，召回率，F1-Score
'''
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

with open(os.path.join(path_pred_result,"Best_model_Power.txt"),"w",encoding = "UTF-8") as f:
    f.write(f"精度(accuracy): \t {round(accuracy,5)}\n")
    f.write(f"查准率(precision): \t {round(precision, 5)}\n")
    f.write(f"召回率(recall): \t {round(recall, 5)}\n")
    f.write(f"F1 Score: \t\t {round(f1, 5)}\n")

'''
混淆矩阵
'''

dataset_label = config["dataset_label"]

confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(dataset_label))])

plt.matshow(confusion, cmap=plt.cm.Oranges)  # Greens, Blues, Oranges, Reds

# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["font.size"] = 8
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

plt.colorbar()
for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.xticks(range(len(dataset_label)), dataset_label, rotation=270)
plt.yticks(range(len(dataset_label)), dataset_label)
plt.title("Confusion Matrix")


plt.savefig(os.path.join(path_pred_result,"Confusion_Matrix.png"))
plt.show()
