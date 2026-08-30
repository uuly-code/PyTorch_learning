import torch

image=torch.rand(1,1,4,4)

model=torch.nn.Sequential( #把多个网络层按顺序连接
    torch.nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        padding=1
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),  #展平
    torch.nn.Linear(8,2)  #全连接层，输入8个特征，输出2个结果
)

output=model(image)

print("输入形状：",image.shape)
print("输出形状：",output.shape)
print("输出结果：",output)


label=torch.tensor([1])#类别编号

loss_fn=torch.nn.CrossEntropyLoss()

loss=loss_fn(output,label)

print("模型输出：",output)
print("正确标签：",label)
print("分类损失：",loss.item())
