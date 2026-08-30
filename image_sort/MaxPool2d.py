import torch

x=torch.tensor([[
    [
        [2,4,7,1],
        [3,5,2,8],
        [9,5,3,4],
        [2,7,0,1]
    ]
]])
"""
第一层[2,4,7,1]表示图片的宽度为4
第二层[
        [2,4,7,1],
        [3,5,2,8],
        [9,5,3,4],
        [2,7,0,1]
    ]表示四行组成一张二维特征图，高4宽4
第三层[[[....]]]加入通道维数，这里是一个通道
第四层[[[[....]]]]加入批量维度，这里表示一张照片
"""
pool=torch.nn.MaxPool2d(kernel_size=2)  #表示2×2的窗口

output=pool(x)

print("输入形状：",x.shape)
print(x)

print("池化后的形状：",output.shape)
print(output)