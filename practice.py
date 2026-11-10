import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 如需绘制损失/准确率曲线，取消下一行注释（需提前安装: pip install matplotlib）
# import matplotlib.pyplot as plt

# ===================== 1. 基础配置 =====================
# 自动选择设备：有GPU用GPU，没有用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 超参数（可自行调整感受效果）
batch_size = 64       # 每批训练的样本数量
learning_rate = 0.001 # 学习率
epochs = 5            # 训练总轮数
input_size = 784      # 输入维度：MNIST图片为28×28=784像素
hidden1_size = 256    # 第一层隐藏层神经元数
hidden2_size = 128    # 第二层隐藏层神经元数
num_classes = 10      # 输出类别数：手写数字0-9共10类

# ===================== 2. 数据集加载 =====================
# 数据预处理：转张量 + 标准化
transform = transforms.Compose([
    transforms.ToTensor(),       # 将图片转为[0,1]范围的张量
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的全局均值与标准差
])

# 自动下载并加载MNIST手写数字数据集
train_dataset = datasets.MNIST(
    root="./data",    # 数据集本地保存路径
    train=True,       # 加载训练集
    transform=transform,
    download=True     # 自动下载，已下载会自动跳过
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,      # 加载测试集
    transform=transform,
    download=True
)

# 数据加载器：分批读取、打乱训练集顺序
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================== 3. 三层神经网络定义 =====================
class ThreeLayerNN(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(ThreeLayerNN, self).__init__()
        # 第一层：输入层 → 隐藏层1（全连接层）
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.relu1 = nn.ReLU()  # ReLU激活函数，引入非线性表达能力
        
        # 第二层：隐藏层1 → 隐藏层2（全连接层）
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.relu2 = nn.ReLU()
        
        # 第三层：隐藏层2 → 输出层（全连接层）
        self.fc3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        # 前向传播：数据从输入到输出的计算流程
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 实例化模型并迁移到对应设备
model = ThreeLayerNN(input_size, hidden1_size, hidden2_size, num_classes).to(device)

# ===================== 4. 损失函数与优化器 =====================
# 交叉熵损失：分类任务的标准损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，收敛稳定，新手友好
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===================== 5. 单轮训练函数 =====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # 切换为训练模式
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        # 将28×28图片展平为784维向量，并迁移到对应设备
        images = images.reshape(-1, input_size).to(device)
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
    
    # 返回本轮平均损失
    return total_loss / len(loader)

# ===================== 6. 测试函数 =====================
def test_epoch(model, loader, device):
    model.eval()  # 切换为评估模式
    correct = 0
    total = 0
    
    # 评估时不计算梯度，节省内存、提升速度
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            # 取概率最大的类别作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 返回分类准确率
    return 100 * correct / total

# ===================== 7. 主程序入口 =====================
if __name__ == "__main__":
    loss_history = []
    acc_history = []
    
    print("开始训练...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = test_epoch(model, test_loader, device)
        
        loss_history.append(train_loss)
        acc_history.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {train_loss:.4f} | 测试准确率: {test_acc:.2f}%")
    
    # 可选：绘制损失和准确率曲线
    # plt.figure(figsize=(10,4))
    # plt.subplot(1,2,1)
    # plt.plot(loss_history)
    # plt.title("Training Loss")
    # plt.subplot(1,2,2)
    # plt.plot(acc_history, color='orange')
    # plt.title("Test Accuracy")
    # plt.show()
    
    # 保存模型权重
    torch.save(model.state_dict(), "three_layer_nn.pth")
    print("\n训练完成！模型权重已保存为 three_layer_nn.pth")
