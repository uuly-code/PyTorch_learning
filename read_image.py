import torch
from PIL import Image
from torchvision import transforms

#使用PIL工具打开图片
image=Image.open("cute-golden-puppy.jpg")

#转换为RGB彩色图片
image=image.convert("RGB")

print("PIL图片大小:",image.size)
print("PIL图片模式:",image.mode)

#定义图片处理步骤
transform=transforms.Compose([  #compose把多个处理步骤按顺序组合
    transforms.Resize((224,224)), #把原始图片调整到224×224
    transforms.ToTensor()         #转换成张量[通道，高，宽]
])

#执行图片处理
image_tensor=transform(image)

print("张量形状：",image_tensor.shape)
print("张量类型：",image_tensor.dtype)
print("最小值：",image_tensor.min().item())
print("最大值：",image_tensor.max().item())

batch_image=image_tensor.unsqueeze(0)
print("增加批量维度后：",batch_image.shape)
