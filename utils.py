'''
2.2.2 生成数据加载器
'''
import random
import os
import yaml
import torch
from PIL import Image

import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset

# 数据归一化与标准化

# 从yaml文件中读
pathYaml_meanVariance = os.path.join("information","meanVariance.yaml")
with open(pathYaml_meanVariance, 'r') as file:
    meanVariance = yaml.load(file, Loader=yaml.FullLoader)



transform_BZ= transforms.Normalize(
    mean=meanVariance["mean"],# 取决于数据集
    std=meanVariance["variance"]
)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        # self.img_size = 512
        self.img_size = 256

        # 操作数据集，格式转换
        self.train_tf = transforms.Compose([
                transforms.Resize(self.img_size),#压缩图片
                transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                # transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.RandomRotation(10, expand=False, center=None),# 随机旋转 -10~10度
                #一半的概率进行3/4裁剪
                transforms.RandomCrop(size=192+64*random.choice([0,1]), padding=None, pad_if_needed=True),
                transforms.ToTensor(),
                transforms.ToPILImage(),
                #亮度、对比度、饱和度 进行随机轻微变换
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),#把图片改为Tensor格式
                transform_BZ#图片标准化
            ])

        self.val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transform_BZ#标准化操作
            ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 读取每一行，返回列表
            imgs_info = f.readlines()
            # 分离：左边是路径，右边是标签
            # lambda匿名函数，x是形参，：后是函数体，最后是传入实参，列表则依次执行
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info#返回图片信息

    def padding_black(self, img):   # 如果尺寸太小可以扩充，填充黑色
        w, h  = img.size
        scale = self.img_size / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = self.img_size
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):#返回真正想返回的东西
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)#打开图片
        img = img.convert('RGB')#转换为RGB 格式
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)

def WriteData(fname, *args):
    with open(fname, 'a+') as f:
        for data in args:
            f.write(str(data)+"\t")
        f.write("\n")


if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print("image.shape = ", image.shape)
        # print("image = ",image)
        print("label = ",label)

# image.shape =  torch.Size([6, 3, 256, 256])
# label =  tensor([1, 7, 4, 6, 7, 4])