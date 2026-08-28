import torch
from torch.utils.data import TensorDataset,DataLoader

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=torch.nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            padding=1
        )
        self.relu=torch.nn.ReLU()
        self.pool=torch.nn.MaxPool2d(kernel_size=2)
        self.flatten=torch.nn.Flatten()
        self.fc=torch.nn.Linear(8,2)

    def forward(self,x):
        #在forward（）中打印每一层的张量形状，学习如何调试复杂模型
        #print("输入：",x.shape)
        x=self.conv(x)
        #print("卷积后：", x.shape)
        x=self.relu(x)
        #print("ReLU后：", x.shape)
        x=self.pool(x)
        #print("池化后：", x.shape)
        x=self.flatten(x)
        #print("展平后：", x.shape)
        x=self.fc(x)
        #print("全连接后：", x.shape)

        return x
train_images = torch.stack([
    torch.full((1, 4, 4), 0.0),
    torch.full((1, 4, 4), 0.1),
    torch.full((1, 4, 4), 0.2),
    torch.full((1, 4, 4), 0.3),

    torch.full((1, 4, 4), 0.7),
    torch.full((1, 4, 4), 0.8),
    torch.full((1, 4, 4), 0.9),
    torch.full((1, 4, 4), 1.0)
])

train_labels = torch.tensor([
    0, 0, 0, 0,
    1, 1, 1, 1
])
train_dataset=TensorDataset(train_images,train_labels)
train_loader=DataLoader(train_dataset,batch_size=2,shuffle=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleCNN().to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

for epoch in range(1, 201):

    model.train()

    for batch_images, batch_labels in train_loader:

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_images)
        loss = loss_fn(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("模型设备：", next(model.parameters()).device)
print("图片设备：", batch_images.device)
print("标签设备：", batch_labels.device)
print(
    "CUDA是否可用：",
    torch.cuda.is_available()
)

print(
    "PyTorch的CUDA版本：",
    torch.version.cuda
)

print(
    "检测到的GPU数量：",
    torch.cuda.device_count()
)