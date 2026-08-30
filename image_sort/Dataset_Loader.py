import torch
from torch.utils.data import TensorDataset,DataLoader

images=torch.stack([
    torch.zeros(1,4,4),
    torch.full((1,4,4),0.1),
    torch.full((1,4,4),0.9),
    torch.ones(1,4,4)
])

labels=torch.tensor([0,0,1,1])

dataset=TensorDataset(images,labels)  #按第0维把图片和标签配对

#查看第1组数据
"""image,label=dataset[0]

print("第一张图片：")
print(image)
print("第一张图片的标签：",label)"""

dataloader=DataLoader(
    dataset,
    batch_size=2,  #每次取出2张照片
    shuffle=True   #每轮数据前打乱数据顺序
)
#查看每一批数据
"""for batch_number,(batch_images,batch_labels) in enumerate(dataloader):
    print(f"第{batch_number}批：")
    print("图片形状：",batch_images.shape)
    print("标签内容：",batch_labels)"""

model=torch.nn.Sequential(
    torch.nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        padding=1
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),
    torch.nn.Linear(8,2)
)

loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(
    model.parameters(),  #需要优化器修改的参数
    lr=0.1  #每次修改的步子大小
)

model.train()

for epoch in range(201):
    total_loss=0

    for batch_images,batch_labels in dataloader:

        outputs=model(batch_images)
        loss=loss_fn(outputs,batch_labels)
        optimizer.zero_grad() #清空旧梯度
        loss.backward()
        optimizer.step() #负责计算怎么修改
        total_loss+=loss.item() #真正手动修改
        #.item()把张量取成普通的数字

""" if epoch%20==0:
        print(
            "训练轮次：",epoch,
            "本轮总损失：",total_loss
        )
"""
model.eval() #评估模式，只预测不训练

correct=0
total=0

dark_image=torch.full((1,1,4,4),0.05)
bright_image=torch.full((1,1,4,4),0.95)

with torch.no_grad(): #只预测，不记录梯度
    dark_output=model(dark_image)
    bright_output=model(bright_image)

    dark_class=dark_output.argmax(dim=1).item()
    bright_class=bright_output.argmax(dim=1).item()

    for batch_images,batch_labels in dataloader:
        outputs=model(batch_images)
        predictions=outputs.argmax(dim=1)
        correct+=(predictions==batch_labels).sum().item()
        total+=batch_labels.size(0)
accuracy=correct/total

print("较暗图片的预测类别：",dark_class)
print("较亮图片的预测类别：",bright_class)
print("预测正确数量：",correct)
print("图片总数：",total)
print("分类准确率：",accuracy)
print("分类准确率：",accuracy*100,"%")


