# 3.5: 接着3.3，提取出判断错误的图片
import os
save_root = "output"
path_pred_result = os.path.join(save_root,"pred_result")
path_pred_result_csv = os.path.join(path_pred_result, "pred_result.csv")


import pandas as pd  # pip install pandas
import os
import cv2
import yaml

pathConfigYaml = os.path.join("information", "config.yaml")
with open(pathConfigYaml, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

myNet = config["myNet"]
dataset_label = config["dataset_label"]
classNum = len(dataset_label)

WrongImgPath = "output\\WrongResult"

if not os.path.isdir(WrongImgPath):
    os.makedirs(WrongImgPath)

pathTestset = os.path.join("datasetGroup", "testGroup.txt")
print(f"Now finding wrong images in {pathTestset}\n")

target_data = pd.read_csv(pathTestset, sep="\t", names=["loc", "type"])
true_label = [i for i in target_data["type"]]

predict_loc = path_pred_result_csv  # 3.ModelEvaluate.py生成的文件

predict_data = pd.read_csv(predict_loc, encoding="GBK")  # ,index_col=0)

predict_label = predict_data.to_numpy().argmax(axis=1)

predict_score = predict_data.to_numpy().max(axis=1)

# print(predict_label)
# print(true_label)
# print(wrong_info[0])
# print(pathTestset)
with open(pathTestset, 'r') as file:
    lines = file.readlines()
    lines = [line.split("\t")[0] for line in lines]
    # print(lines)

for i in range(len(true_label)):
    if not predict_label[i] == true_label[i]:
        print(lines[i], "\tTrue: ",dataset_label[true_label[i]], "\tPredict: ",dataset_label[predict_label[i]])
        # 读取图像
        image = cv2.imread(lines[i])

        # 设置字体、位置、大小、颜色和粗细
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20, 30)  # 文字左下角的位置
        position2 = (20, 50)  # 文字左下角的位置
        font_size = 0.5
        font_color = (0, 0, 255)  # BGR格式，这里是红色
        font_thickness = 1
        outline_font_size = 0.5
        outline_font_color = (255, 255, 255)
        outline_font_thickness = 3
        # 绘制文字(先描边)
        cv2.putText(image, 'True label: ' + dataset_label[true_label[i]], position, font, outline_font_size,
                    outline_font_color,
                    outline_font_thickness)
        cv2.putText(image, 'True label: ' + dataset_label[true_label[i]], position, font, font_size, font_color,
                    font_thickness)
        cv2.putText(image, 'Predicted:  ' + dataset_label[predict_label[i]], position2, font, outline_font_size,
                    outline_font_color,
                    outline_font_thickness)
        cv2.putText(image, 'Predicted:  ' + dataset_label[predict_label[i]], position2, font, font_size, font_color,
                    font_thickness)

        # cv2.imshow('Image with Text', image)
        # cv2.waitKey(0)
        # break

        tempPath = os.path.join(WrongImgPath, lines[i].split("\\")[-1])
        # print(tempPath)
        cv2.imencode('.jpg', image)[1].tofile(tempPath)  # 保存图片
