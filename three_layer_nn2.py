# -*- coding: utf-8 -*-
"""
three_layer_nn.py —— 三层神经网络识别手写数字（MNIST）
================================================================================

【任务】
    给一张 28×28 的手写数字图片，判断它是 0~9 中的哪一个数字。

【三层】
    输入层(784) -> 隐藏层1(256) -> 隐藏层2(128) -> 输出层(10)
    真正带参数的是最后三个 nn.Linear（两个隐藏层 + 一个输出层）。

【运行】
    python three_layer_nn.py

    第一次运行会自动下载 MNIST 数据集（约 11MB）到 ./data 目录，
    以后直接读本地文件，不再联网。

【输出】
    每轮打印训练损失和测试准确率，5 轮后测试准确率约 97%~98%；
    训练完把模型权重保存为 mnist_mlp.pth。

【结构】
    1. 基础配置    2. 数据集加载    3. 网络定义
    4. 损失与优化器 5. 训练函数      6. 测试函数      7. 主程序入口
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 如需绘制损失/准确率曲线，取消下一行注释（需提前安装: pip install matplotlib）
# import matplotlib.pyplot as plt


# ===================== 1. 基础配置 =====================
# 自动选择设备：有 GPU 用 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 固定随机种子：让每次运行的结果可复现（删掉也能跑，只是每次会有一点波动）
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 超参数（可自行调整感受效果）
batch_size = 64               # 每批训练的样本数量
learning_rate = 0.001         # 学习率：每次参数更新的步子大小
epochs = 5                    # 训练总轮数
input_size = 784              # 输入维度：MNIST 图片 28×28 = 784
hidden1_size = 256            # 第一层隐藏层神经元数
hidden2_size = 128            # 第二层隐藏层神经元数
num_classes = 10              # 输出类别数：手写数字 0~9 共 10 类
model_path = "mnist_mlp.pth"  # 训练完把权重存到这个文件


# ===================== 2. 数据集加载 =====================
# 预处理流水线：把图片转成张量，再做标准化
transform = transforms.Compose([
    transforms.ToTensor(),                       # 图片 -> [0,1] 的张量，形状 (1,28,28)
    transforms.Normalize((0.1307,), (0.3081,)),  # 减均值、除标准差（MNIST 的固定数值）
])

# 训练集：6 万张。download=True 表示第一次运行自动下载，已下载会自动跳过
train_dataset = datasets.MNIST(
    root="./data",        # 数据集本地保存路径
    train=True,           # True = 训练集
    transform=transform,  # 套用上面的预处理
    download=True,
)

# 测试集：1 万张。预处理必须和训练集完全一致，否则准确率没有意义
test_dataset = datasets.MNIST(
    root="./data",
    train=False,          # False = 测试集
    transform=transform,
    download=True,
)

# 数据加载器：分批读取；训练集每轮打乱顺序，测试集不需要
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ===================== 3. 三层神经网络定义 =====================
class ThreeLayerNN(nn.Module):
    """输入 784 -> 隐藏层 256 -> 隐藏层 128 -> 输出 10"""

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        # 第一层：输入 -> 隐藏层1
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.relu1 = nn.ReLU()
        # 第二层：隐藏层1 -> 隐藏层2
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.relu2 = nn.ReLU()
        # 第三层：隐藏层2 -> 输出层（10 个打分，后面接交叉熵，所以不加 softmax）
        self.fc3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        # 前向传播：数据从输入到输出的计算流程
        x = x.reshape(x.size(0), -1)  # (N,1,28,28) -> (N,784)，把每张图摊平
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# 实例化模型并迁移到对应设备
model = ThreeLayerNN(input_size, hidden1_size, hidden2_size, num_classes).to(device)


# ===================== 4. 损失函数与优化器 =====================
# 交叉熵损失：分类任务的标准损失函数，内部自带 softmax
criterion = nn.CrossEntropyLoss()
# Adam 优化器：自适应学习率，收敛稳定，新手友好
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ===================== 5. 单轮训练函数 =====================
def train_epoch(model, loader, criterion, optimizer, device):
    """把训练集完整跑一遍，返回平均损失。"""
    model.train()  # 切换为训练模式
    total_loss = 0.0

    for images, labels in loader:
        # 数据搬到和模型同一个设备上
        images = images.to(device)
        labels = labels.to(device)

        # ① 前向传播：计算模型预测值
        outputs = model(images)
        # ② 计算损失：预测值与真实标签的差距
        loss = criterion(outputs, labels)

        # ③ 反向传播 + 参数更新
        optimizer.zero_grad()  # 清空上一轮梯度（避免累加）
        loss.backward()        # 反向传播：自动计算所有参数的梯度
        optimizer.step()       # 根据梯度更新网络参数

        total_loss += loss.item()

    return total_loss / len(loader)


# ===================== 6. 测试函数 =====================
def test_epoch(model, loader, device):
    """在测试集上算准确率，返回百分数（比如 97.20）。"""
    model.eval()  # 切换为评估模式
    correct = 0
    total = 0

    # 评估时不计算梯度，节省内存、提升速度
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)  # 每行最大值的下标 = 预测的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ===================== 7. 主程序入口 =====================
if __name__ == "__main__":
    loss_history = []  # 记录每轮的训练损失（画图用）
    acc_history = []   # 记录每轮的测试准确率（画图用）

    print("开始训练...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = test_epoch(model, test_loader, device)

        loss_history.append(train_loss)
        acc_history.append(test_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {train_loss:.4f} | 测试准确率: {test_acc:.2f}%")

    # 可选：绘制损失和准确率曲线（把下面几行前面的 # 删掉即可）
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_history)
    # plt.title("Training Loss")
    # plt.subplot(1, 2, 2)
    # plt.plot(acc_history, color="orange")
    # plt.title("Test Accuracy")
    # plt.show()

    # 保存模型权重
    torch.save(model.state_dict(), model_path)
    print(f"\n训练完成！模型权重已保存为 {model_path}")
