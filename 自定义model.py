import torch

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

model=SimpleCNN()

image=torch.rand(1,1,4,4)
output=model(image) #PyTorch自动调用该模型的forward

