import torch
#from torchvision import datasets   #ImageFolder就在里面
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#训练集：允许随机变化
train_transform=transforms.Compose([
    transforms.Resize((64,64)),  #把所有图片都调整为64×64

    transforms.RandomHorizontalFlip(p=0.5),  #50%的概率左右翻转
    transforms.RandomRotation(degrees=10),   #在-10度到+10度之间随机旋转

    transforms.ToTensor(),       #转换为PyTorch张量 变成[3,64,64]
])

#验证集：只调整大小，不随机变化
val_transform=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataset=datasets.ImageFolder(root="image_dataset/train",transform=train_transform)
val_dataset=datasets.ImageFolder(root="image_dataset/val",transform=train_transform)
#ImageFolder是根据子文件名称确定类别

"""
for i in range(6):
    image,label=dataset[0]

    show_image=image.permute(1,2,0) #重新排列三个维度[高，宽，通道]

    plt.subplot(2,3,i+1)  #把六张图片排列成2行3列
    plt.imshow(show_image)
    plt.title(dataset.classes[label])
    plt.axis("off")

plt.tight_layout()
plt.show()
"""

train_loader=DataLoader(
    train_dataset,batch_size=2,shuffle=True
)
val_loader=DataLoader(
    val_dataset,batch_size=2,shuffle=False
)
print("训练图片数量：", len(train_dataset))
print("验证图片数量：", len(val_dataset))
print("训练类别：", train_dataset.classes)
print("验证类别：", val_dataset.classes)


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

        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)

        x=self.flatten(x)
        x=self.fc(x)

        return x

model=RealImageCNN()

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

train_loss_history=[]
val_loss_history=[]
train_acc_history=[]
val_acc_history=[]

#开始训练
for epoch in range(1,101):
    model.train()

    train_loss=0
    train_correct=0
    train_total=0

    for batch_images,batch_labels in train_loader:
        outputs=model(batch_images)

        loss=loss_fn(outputs,batch_labels)
        optimizer.zero_grad()  #清空旧梯度
        loss.backward()        #反向传播(自动计算梯度
        optimizer.step()       #更新模型参数
        train_loss+=loss.item()
        predictions=outputs.argmax(dim=1)

        train_correct+=(predictions==batch_labels).sum().item()
        train_total+=batch_labels.size(0)

        train_accuracy=train_correct/train_total*100


    model.eval()
    val_loss=0
    val_correct=0
    val_total=0

    with torch.no_grad():
        for batch_images,batch_labels in val_loader:
            outputs=model(batch_images)

            loss=loss_fn(outputs,batch_labels)
            val_loss+=loss.item()

            predictions=outputs.argmax(dim=1)

            val_correct+=(predictions==batch_labels).sum().item()
            val_total+=batch_labels.size(0)

            val_accuracy=val_correct/val_total*100


   #计算损失并保存结果
    average_train_loss=train_loss/len(train_loader)
    average_val_loss=val_loss/len(val_loader)

    train_loss_history.append(average_train_loss)
    val_loss_history.append(average_val_loss)

    train_acc_history.append(train_accuracy)
    val_acc_history.append(val_accuracy)

    if epoch%10==0:

        average_train_loss=train_loss/len(train_loader)
        average_val_loss=val_loss/len(val_loader)
        print(
            "epoch:",epoch,
            "train_loss:",average_train_loss,"train_accuracy:",train_accuracy,"%",
            "val_loss:",average_val_loss,"val_accuracy:",val_accuracy,"%")

epochs=range(1,101)
plt.figure(figsize=(10,4))

#left:损失曲线
plt.subplot(1,2,1)  #1行2列的第一个位置
plt.plot(epochs,train_loss_history,label="train loss")
plt.plot(epochs,val_loss_history,label="val loss")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Curve")
plt.legend()

#right:准确率曲线
plt.subplot(1,2,2)  #1行2列的第二个位置
plt.plot(epochs,train_acc_history,label="train accuracy")
plt.plot(epochs,val_acc_history,label="val accuracy")

plt.xlabel("epoch")
plt.ylabel("accuracy(%)")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()