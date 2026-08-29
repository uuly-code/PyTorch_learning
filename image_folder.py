import torch
from torchvision import datasets   #ImageFolder就在里面
from torchvision import transforms #导入图片处理工具
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.Resize((64,64)),  #把所有图片都调整为64×64
    transforms.ToTensor(),       #转换为PyTorch张量 变成[3,64,64]
])

dataset=datasets.ImageFolder(root="image_dataset",transform=transform)
#ImageFolder是根据子文件名称确定类别

print("图片总数：", len(dataset))
print("类别名称：", dataset.classes)
print("类别编号：", dataset.class_to_idx)

"""
image,label=dataset[0]
print("第一张图片形状：",image.shape)
print("第一张图片标签：",label)
"""

dataloader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

class RealImageCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=torch.nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.relu1=torch.nn.ReLU()
        self.pool1=torch.nn.MaxPool2d(kernel_size=2)
        self.conv2=torch.nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten=torch.nn.Flatten()
        self.fc=torch.nn.Linear(16*16*16,2)

    def forward(self,x):

        #print("输入：",x.shape)

        x=self.conv1(x)
        #print("第一次卷积后：",x.shape)

        x=self.relu1(x)
        x=self.pool1(x)
        #print("第一次池化后：",x.shape)

        x=self.conv2(x)
        #print("第二次卷积后：",x.shape)

        x=self.relu2(x)
        x=self.pool2(x)
        #print("第二次池化后：",x.shape)

        x=self.flatten(x)
        #print("展平后：", x.shape)

        x=self.fc(x)
        #print("全连接后：", x.shape)

        return x

model=RealImageCNN()

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

#开始训练
for epoch in range(1,101):
    model.train()

    total_loss=0
    correct=0
    total=0

    for batch_images,batch_labels in dataloader:
        outputs=model(batch_images)

        loss=loss_fn(outputs,batch_labels)
        optimizer.zero_grad()  #清空旧梯度
        loss.backward()        #反向传播
        optimizer.step()       #更新模型参数
        total_loss+=loss.item()
        predictions=outputs.argmax(dim=1)

        correct+=(predictions==batch_labels).sum().item()
        total+=batch_labels.size(0)

    accuracy=correct/total*100

    if epoch%10==0:
        print("epoch:",epoch,"loss:",total_loss,"accuracy:",accuracy,"%")

model.eval()
with torch.no_grad():
    for batch_images,batch_labels in dataloader:
        outputs=model(batch_images)

        probabilities=torch.softmax(outputs,dim=1)
        predictions=outputs.argmax(dim=1)

        print("类别分数：",outputs)
        print("类别概率：",probabilities)
        print("预测类别：",predictions)
        print("正确类别：",batch_labels)



"""
#逐批读取图片
for batch_number,(batch_image,batch_labels) in enumerate(dataloader):
    print("批次编号：",batch_number)
    print("图片形状：",batch_image.shape)
    print("图片标签：",batch_labels)
"""