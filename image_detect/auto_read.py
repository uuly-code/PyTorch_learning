import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#定义检测数据集
class DefectDetectionDataset(Dataset):
    def __init__(self,images_dir,label_dir):
        self.images_dir=images_dir  #路径只写到文件夹
        self.labels_dir=label_dir

        self.image_names=sorted([
            name  #image_names=[]
            for name in os.listdir(images_dir) #文件夹路径
            if name.endswith((".png",".jpg",".jpeg")) #image_names.append(name)
        ])
        # self.image_names=sorted(image_names)
        self.to_tensor=transforms.ToTensor()

    def __len__(self):   #这个数据集一共有多少张照片
        return len(self.image_names)

    def __getitem__(self,index):  #读取指定的一组数据

        image_name=self.image_names[index]  #获取图片的文件名

        image_path=os.path.join(    #拼接完整图片路径
            self.images_dir,
            image_name
        )

        label_name=os.path.splitext(image_name)[0]+".txt"  #根据图片名得到标签名

        label_path=os.path.join(
            self.labels_dir,
            label_name
        )

        image=Image.open(image_path).convert("RGB")
        image_width,image_height=image.size

        boxes=[]
        labels=[]
        if os.path.exists(label_path): #如果文件存在就打开
            with open(label_path,"r") as file:#读取标签文件中的所有框
                for line in file:

                    if line.strip()=="":    #跳过空白行
                        continue

                    data=line.split()
                    class_id=int(data[0])
                    x_center=float(data[1])*image_width
                    y_center=float(data[2])*image_height
                    box_width=float(data[3])*image_width
                    box_height=float(data[4])*image_height

                    x_min=x_center-box_width/2
                    y_min=y_center-box_height/2
                    x_max=x_center+box_width/2
                    y_max=y_center+box_height/2

                    boxes.append([x_min,y_min,x_max,y_max])

                    labels.append(class_id)

        if random.random()<0.5:  #50%的概率进行水平翻转

            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)#左右翻转图片

            for box in boxes: #同步修改每个框的横坐标
                old_x_min=box[0]
                old_x_max=box[2]

                box[0]=image_width-old_x_max
                box[2]=image_width-old_x_min

        target_width=640
        target_height=640

        #计算横向和纵向的缩放比例
        scale_x=target_width/image_width
        scale_y=target_height/image_height

        #缩放图片
        image=image.resize((target_width,target_height))

        for box in boxes:  #同步缩放所有框
            box[0]=box[0]*scale_x
            box[1]=box[1]*scale_y
            box[2]=box[2]*scale_x
            box[3]=box[3]*scale_y
        #转换为PyTorch张量并返回
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        labels=torch.tensor(labels,dtype=torch.int64)

        image=self.to_tensor(image)

        target={"boxes":boxes,"labels":labels}
        return image,target
def collate_fn(batch):
    images=[]
    targets=[]

    for image,target in batch:
        images.append(image)
        targets.append(target)

    images=torch.stack(images,dim=0)  #把同尺寸图片合成一个批量张量

    return images,targets

#创建数据集
dataset=DefectDetectionDataset(
    images_dir="detection_dataset/images",
    label_dir="detection_dataset/labels"
)

image,target=dataset[0]
#创建DataLoader
dataloader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

class_names={0:"scretch",1:"crack"}
for index in range(len(dataset)):
    image,target=dataset[index]

    show_image=image.permute(1,2,0)  #[通道,高,宽]-->>[高,宽,通道]

    fig,ax=plt.subplots()

    ax.imshow(show_image)
    ax.set_title(dataset.image_names[index])
    ax.axis("off")

    boxes=target["boxes"]
    labels=target["labels"]

    for box,label in zip(boxes,labels):  #把框和类一一对应
        x_min=box[0].item()
        y_min=box[1].item()
        x_max=box[2].item()
        y_max=box[3].item()

        box_width=x_max-x_min
        box_height=y_max-y_min

        rectangle=patches.Rectangle(  #创建矩形框
            (x_min,y_min),  #框的起点，也就是框的左上角
            box_width,
            box_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"  #内部不填充颜色
        )

        ax.add_patch(rectangle)

        class_id=label.item()
        class_name=class_names[class_id]

        ax.text(
            x_min,
            y_min-10, #字体位置
            class_name,
            color="red",
            fontsize=12  #字体大小
        )
    plt.show()
"""
print("图片形状：",image.shape)
print("缺陷框：",target["boxes"])
print("类别编号：",target["labels"])
"""

for images,targets in dataloader:
    print("这一批图片的数量：",len(images))
    print("这一批图片的形状：",images.shape)
    print("这一批标签的数量：",len(targets))

    #print("第一张图片形状：",images[0].shape)
    print("第一张图片的框：",targets[0]["boxes"])
    #print("第一张图片形状：",targets[0]["labels"])

    #print("第二张图片形状：",images[1].shape)
    print("第二张图片的框：",targets[1]["boxes"])
    #print("第二张图片形状：",targets[1]["labels"])

    break


for index in range(len(dataset)):

    image, target = dataset[index]

    print("编号：", index)
    print("图片名：", dataset.image_names[index])
    print("图片形状：", image.shape)
    print("缺陷框：", target["boxes"])
    print("类别：", target["labels"])
    print()