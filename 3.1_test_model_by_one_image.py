'''
    单图测试
'''

import torch
import yaml
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
from PIL import Image
import torchvision.transforms as transforms
import os

# 从yaml文件中读
pathYaml_meanVariance = os.path.join("information", "meanVariance.yaml")
with open(pathYaml_meanVariance, 'r') as file:
    meanVariance = yaml.load(file, Loader=yaml.FullLoader)
pathConfigYaml = os.path.join("information", "config.yaml")
with open(pathConfigYaml, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
myNet = config["myNet"]
dataset_label = config["dataset_label"]
classNum = len(dataset_label)

transform_BZ = transforms.Normalize(
    mean=meanVariance["mean"],  # 取决于数据集
    std=meanVariance["variance"]
)


def padding_black(img, img_size=256):  # 如果尺寸太小可以扩充
    w, h = img.size
    scale = img_size / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = img_size
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img


if __name__ == '__main__':

    # img_path = r'dataset\Rainy\h_a7.png'
    img_path = r'dataset\Asphalt\a_7.png'

    val_tf = transforms.Compose([  ##简单把图片压缩了变成Tensor模式
        # transforms.Resize(256),
        transforms.ToTensor(),
        transform_BZ  # 标准化操作
    ])

    # 如果显卡可用，则用显卡
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2
    if myNet == "resnet18":
        finetune_net = resnet18(num_classes=classNum)
    elif myNet == "resnet34":
        finetune_net = resnet34(num_classes=classNum)
    elif myNet == "resnet50":
        finetune_net = resnet50(num_classes=classNum)
    elif myNet == "resnet101":
        finetune_net = resnet101(num_classes=classNum)
    elif myNet == "resnet152":
        finetune_net = resnet152(num_classes=classNum)
    elif myNet == "mobilenet_v2":
        finetune_net = mobilenet_v2(num_classes=classNum)
    else:
        finetune_net = resnet18(num_classes=classNum)
        print(f"不存在模型{myNet},将默认使用resnet18")

    # 自动读取模型名字
    save_root = "output"
    model_name = myNet + "_" + config["mark"]
    model_path = os.path.join(save_root, model_name + "_best.pth")
    # 你也可以手动指定模型
    # model_path = os.path.join(save_root, "resnet18_best.pth")

    # 加载模型
    state_dict = torch.load(model_path)

    # print("state_dict = ", state_dict)
    finetune_net.load_state_dict(state_dict)
    finetune_net.eval()
    finetune_net.to(device)
    with torch.no_grad():

        # finetune_net.to(device)
        img = Image.open(img_path)  # 打开图片
        img = img.convert('RGB')  # 转换为RGB 格式
        img = padding_black(img, 256)
        img = val_tf(img)
        img_tensor = torch.unsqueeze(img, 0)  # N,C,H,W, ; C,H,W 进行扩张，多一个n的通道

        img_tensor = img_tensor.to(device)
        result = finetune_net(img_tensor)
        # print("result = ",result.argmax(1))

        id = result.argmax(1).item()

        print("预测结果为：", dataset_label[id])
