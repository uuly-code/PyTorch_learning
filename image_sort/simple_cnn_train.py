import torch

torch.manual_seed(42)

images=torch.stack([
    torch.zeros(1,4,4),                    #很暗：第0类
    torch.full((1,4,4),0.1),  #较暗：第0类
    torch.full((1,4,4),0.9),  #较亮：第1类
    torch.ones(1,4,4)                      #很亮：第1类
])
#每张图片的正确类别
labels=torch.tensor([0,0,1,1])

print("图片形状：",images.shape)
print("标签形状：",labels.shape)

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

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#重复训练
for epoch in range(501):
    outputs=model(images)
    loss=loss_fn(outputs,labels)

    optimizer.zero_grad()     #清空旧梯度
    loss.backward()           #计算参数应该怎么修改
    optimizer.step()          #真正修改模型参数

    if epoch % 100 == 0:     #每100轮打印一次
        predictions=outputs.argmax(dim=1)
        print(
            "epoch:",epoch,
            "loss:",loss.item(),
            "predictions:",predictions.tolist()
        )

dark_image=torch.full((1,1,4,4),0.05)
bright_image=torch.full((1,1,4,4),0.95)

dark_output=model(dark_image)
bright_output=model(bright_image)

dark_class=dark_output.argmax(dim=1).item()
bright_class=bright_output.argmax(dim=1).item()

print("较暗新照片的预测类别:",dark_class)
print("较亮新照片的预测类别:",bright_class)
