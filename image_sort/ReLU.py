import torch

x=torch.tensor([
    [-2.0,-1.0,0.0],
    [1.0,2.0,3.0]
])

relu=torch.nn.ReLU()
output=relu(x)

print("输入：")
print(x)

print("经过ReLU：")
print(output)