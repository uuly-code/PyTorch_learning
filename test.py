import torch

# 1. 准备数据：规律是 y = 2x
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. 定义一个最简单的模型：输入1个数，输出1个数
model = torch.nn.Linear(1, 1)

# 3. 定义损失函数：预测和答案相差多少
loss_fn = torch.nn.MSELoss()

# 4. 定义优化器：负责调整模型参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. 重复训练
for epoch in range(5000):
    prediction = model(x)          # 做预测
    loss = loss_fn(prediction, y)  # 计算错误

    optimizer.zero_grad()          # 清空上一次的梯度
    loss.backward()                # PyTorch自动求导
    optimizer.step()               # 更新模型参数

weight = model.weight.item()
bias = model.bias.item()

print("学到的权重：", weight)
print("学到的偏置：", bias)

# 6. 看训练后的模型能否推测 5 × 2
test = torch.tensor([[5.0]])
print("模型预测：", model(test).item())
print("正确答案：10")

print("x 的内容：")
print(x)

print("x 的形状：", x.shape)

prediction = model(x)

print("预测结果：")
print(prediction)

print("预测结果的形状：", prediction.shape)