import torch

#一张通道为3，4×4的模拟图片
image = torch.rand(3,4,4)
print("单张图片：",image.shape)

#在最前面增加批量维度,每调用一次只增加一个维度,()里面是新维度的位置
batch=image.unsqueeze(0)
batch_a=image.unsqueeze(0).unsqueeze(0)
print("加入批量维度：",batch.shape,batch_a.shape)

#可以先增加批量维度，再把图片复制3份
batch_b=image.unsqueeze(0).repeat(3,1,1,1)
print("加入批量维度：",batch_b.shape)

#删除大小为1的批量维度，只能删除大小为1的维度,()里的也是位置
single_image=batch.squeeze(1)
print("删除批量维度：",single_image.shape)