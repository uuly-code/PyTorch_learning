from torchvision import datasets   #ImageFolder就在里面
from torchvision import transforms #导入图片处理工具

transform=transforms.Compose([
    transforms.Resize((64,64)),  #把所有图片都调整为64×64
    transforms.ToTensor(),       #转换为PyTorch张量 变成[3,64,64]
])

dataset=datasets.ImageFolder(root="image_dataset",transform=transform)
#ImageFolder是根据子文件名称确定类别

print("图片总数：", len(dataset))
print("类别名称：", dataset.classes)
print("类别编号：", dataset.class_to_idx)

image,label=dataset[0]
print("第一张图片形状：",image.shape)
print("第一张图片标签：",label)