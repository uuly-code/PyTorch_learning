import torch

image=torch.rand(1,3,4,4)

#定义二维卷积层
conv1 = torch.nn.Conv2d(
    in_channels=3,  #输入图片有3个通道RGB
    out_channels=2, #使用2组卷积核，生成2张新的特征图，输出通道数变成2，指代模型提取的两种特征
    kernel_size=3,  #卷积核的大小是3×3
    padding=1,      #再图片周围补一圈像素，使处理后的特征图高和宽不变
    stride=1        #步长
)
conv2 = torch.nn.Conv2d(
    in_channels=3,
    out_channels=2,
    kernel_size=3,
    padding=1,
    stride=2        #缩小了特征图
)

#让图片经过卷积层
output1=conv1(image)
output2=conv2(image)

#print("输入形状：",image.shape)
#print("输出形状：",output1.shape)
#print("输出形状：",output2.shape)


relu=torch.nn.ReLU()  #不改变张量形状，只改变里面的数值
feature=conv1(image)
output=relu(feature)

print("卷积前：",image.shape)
print("卷积后：",feature.shape)
print("ReLU后：",output.shape)