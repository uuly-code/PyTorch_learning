import torch
from torch.utils.data import TensorDataset,DataLoader

#训练集
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
#验证集
val_images = torch.stack([
    torch.full((1, 4, 4), 0.12),
    torch.full((1, 4, 4), 0.28),
    torch.full((1, 4, 4), 0.72),
    torch.full((1, 4, 4), 0.88)
])

val_labels = torch.tensor([0, 0, 1, 1])

#测试集
test_images = torch.stack([
    torch.full((1, 4, 4), 0.15),
    torch.full((1, 4, 4), 0.25),
    torch.full((1, 4, 4), 0.75),
    torch.full((1, 4, 4), 0.85)
])

test_labels=torch.tensor([0,0,1,1])

train_dataset=TensorDataset(train_images,train_labels)
val_dataset = TensorDataset(val_images,val_labels)
test_dataset=TensorDataset(test_images,test_labels)


train_loader=DataLoader(train_dataset,batch_size=2,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=2,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=2,shuffle=False)


model = torch.nn.Sequential(
    torch.nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        padding=1
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),
    torch.nn.Linear(8, 2)
)

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

best_val_accuracy=0  #最佳验证正确率

for epoch in range(201):
    model.train()
    train_loss=0
    for batch_images, batch_labels in train_loader:
        outputs=model(batch_images)
        loss=loss_fn(outputs,batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()


    if epoch % 20 ==0:
        model.eval()

        val_correct=0
        val_total=0
        val_loss=0
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                outputs = model(batch_images)
                loss=loss_fn(outputs,batch_labels)
                predictions = outputs.argmax(dim=1)
                val_loss+=loss.item()
                val_correct+=(predictions==batch_labels).sum().item()
                val_total += batch_labels.size(0)
        val_accuracy=(val_correct/val_total)*100

        print("轮次:",epoch,"训练损失:",train_loss,"验证损失：",val_loss,"验证准确率：",val_accuracy,"%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            #model.state_dict()表示获取模型中已经学到的所有参数
            print("保存了新的最佳模型")

model.load_state_dict(torch.load("best_model.pth", weights_only=True))
#从best_model_pth文件读取之前保存的最佳参数，然后把这些参数装回当前模型，，weights_only=True表示只读取权重等模型参数
print("已加载最佳模型")

model.eval()

test_correct = 0
test_total = 0
test_loss = 0

with torch.no_grad():

    for batch_images, batch_labels in test_loader:
        outputs = model(batch_images)
        loss = loss_fn(outputs, batch_labels)
        predictions = outputs.argmax(dim=1)
        test_loss += loss.item()
        test_correct += (predictions == batch_labels).sum().item()
        test_total += batch_labels.size(0)

        print("模型预测：", predictions)
        print("正确标签：", batch_labels)

test_accuracy = test_correct / test_total * 100

print("测试损失：", test_loss)
print("测试正确数量：", test_correct)
print("测试图片数量：", test_total)
print("测试准确率：", test_accuracy, "%")
