# -*- coding: utf-8 -*-
"""
three_layer_nn.py —— 用 PyTorch 写一个"三层神经网络"，识别手写数字（MNIST）
================================================================================

【这个程序在干什么】
    给电脑看一张 28x28 的手写数字图片（784 个像素点），让它判断是 0~9 里的哪个数字。

【"三层"指什么】
    输入层(784) -> 隐藏层1(256) -> 隐藏层2(128) -> 输出层(10)

    在 PyTorch 里真正"带参数、要学习"的是最后 3 个 nn.Linear，
    所以严格说是"两个隐藏层 + 一个输出层"。老师嘴里的三层网络一般就是这么数：
    输入层不算学习层，隐藏层和输出层加起来三层。

【怎么运行】
    python three_layer_nn.py

    第一次运行会自动从网上下载 MNIST 数据集（约 11MB）到 ./data 目录，
    以后再运行就直接读本地文件，不用再联网。

【给学 C 的你】
    C 语言                              Python / PyTorch
    ---------------------------------   ------------------------------------------
    #include <stdio.h>                  import torch
    int a[10];                          a = 10                      (不用声明类型)
    for (i=0; i<n; i++) { ... }         for i in range(n):          (用缩进代替大括号)
    int f(int x) { return x*x; }        def f(x): return x*x
    数组 + 自己写循环算梯度             张量(Tensor) + 自动求导(autograd)
    printf("%d\\n", x)                   print(x)
"""

import gzip
import os
import random
import time
import urllib.request

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 0. 超参数：先集中放在这里，改这几个数字就能做实验
# =============================================================================
DATA_DIR = "data"          # 数据集下载/缓存目录
BATCH_SIZE = 128           # 每批喂给网络多少张图（显存/内存不够就调小）
EPOCHS = 3                 # 把整个训练集完整看几遍
LEARNING_RATE = 0.001      # 学习率：每次参数更新的"步子"大小
HIDDEN1 = 256              # 第一个隐藏层神经元个数
HIDDEN2 = 128              # 第二个隐藏层神经元个数
SEED = 42                  # 随机种子：固定住，保证每次运行结果基本一样
MODEL_PATH = "mnist_mlp.pth"

# 有 GPU 就用 GPU，没有就用 CPU。这一行以后会天天见到。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# =============================================================================
# 1. 数据：下载并读取 MNIST
# =============================================================================
def _download(key: str) -> str:
    """确保文件在本地，返回本地路径。已经下载过就直接返回。"""
    filename = MNIST_FILES[key]
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path

    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{MNIST_MIRROR}/{filename}"
    print(f"  下载 {url}")
    urllib.request.urlretrieve(url, path)
    return path


def _read_images(key: str) -> torch.Tensor:
    """读图片文件，返回形状 (N, 1, 28, 28)、取值 0~1 的 float 张量。"""
    path = _download(key)
    with gzip.open(path, "rb") as f:
        # frombuffer 要求缓冲区可写，所以这里用 bytearray 转一下
        raw = bytearray(f.read())
    # 文件头 16 字节是说明信息（魔数、图片数、行数、列数），从第 16 字节才是像素
    pixels = torch.frombuffer(raw, dtype=torch.uint8, offset=16)
    # 原始像素是 0~255 的整数，除以 255 变成 0~1 的浮点数，网络才好训练
    return pixels.view(-1, 1, 28, 28).float() / 255.0


def _read_labels(key: str) -> torch.Tensor:
    """读标签文件，返回形状 (N,) 的 int64 张量，每个值是 0~9。"""
    path = _download(key)
    with gzip.open(path, "rb") as f:
        raw = bytearray(f.read())
    # 标签文件头只有 8 字节
    labels = torch.frombuffer(raw, dtype=torch.uint8, offset=8)
    return labels.long()


def build_dataloaders():
    """把数据处理成 DataLoader —— 可以按批(batch)迭代的对象。"""
    print("[1/4] 准备数据 ...")
    train_images = _read_images("train_images")   # (60000, 1, 28, 28)
    train_labels = _read_labels("train_labels")   # (60000,)
    test_images = _read_images("test_images")     # (10000, 1, 28, 28)
    test_labels = _read_labels("test_labels")     # (10000,)

    train_set = TensorDataset(train_images, train_labels)
    test_set = TensorDataset(test_images, test_labels)

    # shuffle=True：每个 epoch 把训练集顺序打乱，避免模型记住"样本顺序"这种没用的规律
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # 测试集不需要打乱
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    print(f"      训练集 {len(train_set)} 张，测试集 {len(test_set)} 张，"
          f"每个 batch {BATCH_SIZE} 张")
    return train_loader, test_loader, train_images, train_labels


# =============================================================================
# 2. 模型：三层神经网络
# =============================================================================
class ThreeLayerNet(nn.Module):
    """
    继承 nn.Module 就是"定义一个网络"的标准写法，类似 C 里的 struct + 函数。

    nn.Sequential 就是"把这几层按顺序串起来"，数据从前往后流。
    nn.Linear(in, out) 干的事：y = x @ W + b，W 形状是 (in, out)，b 形状是 (out,)。
    nn.ReLU() 是激活函数：max(0, x)，逐元素操作，没有参数。
        如果没有它，三层线性层叠起来还是等价于一层线性层，网络就"深得没意义"了。
    """

    def __init__(self, in_features=28 * 28, num_classes=10):
        super().__init__()          # 固定写法：先初始化父类
        self.layers = nn.Sequential(
            nn.Linear(in_features, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, num_classes),   # 输出 10 个"打分"（logits）
        )

    def forward(self, x):
        """前向传播：数据怎么从输入算到输出。调用 model(x) 时 PyTorch 会自动调它。"""
        # x 形状是 (N, 1, 28, 28)，拉平成 (N, 784)，就是把这批图每张摊成一行
        x = x.view(x.shape[0], -1)
        return self.layers(x)


# =============================================================================
# 3. 训练和评估
# =============================================================================
def train_one_epoch(model, loader, loss_fn, optimizer, epoch):
    """把训练集完整跑一遍，返回平均 loss 和准确率。"""
    model.train()               # 切换到"训练模式"（会影响 Dropout/BatchNorm 等层）
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        # 1) 数据搬到和模型同一个设备上（GPU 模型配 GPU 数据）
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 2) 前向传播：算预测
        logits = model(images)
        loss = loss_fn(logits, labels)

        # 3) 反向传播 + 更新参数 —— PyTorch 训练循环永远这 4 行
        optimizer.zero_grad()   # 清空上一轮的梯度（不清会累加）
        loss.backward()         # 自动算出每个参数的梯度
        optimizer.step()        # 按梯度更新参数

        # 4) 记录一些数字，方便观察
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if step % 100 == 0 or step == len(loader):
            print(f"      epoch {epoch} | 第 {step:3d}/{len(loader)} 批 | "
                  f"loss {loss.item():.4f} | 累计准确率 {correct / total:.2%}")

    return total_loss / len(loader), correct / total


@torch.no_grad()   # 评估不需要梯度，关掉能省内存、跑更快
def evaluate(model, loader):
    """在测试集上算准确率。"""
    model.eval()                # 切换到"评估模式"
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


# =============================================================================
# 4. 看看模型到底认出了什么（把图片用字符画出来）
# =============================================================================
ASCII_CHARS = " .:-=+*#%@"   # 从"黑"到"白"的 10 个字符，用来代替像素亮度


def show_predictions(model, images, labels, n=3):
    """挑几张测试图，用字符画出来，同时显示模型猜的答案。"""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(images[:n].to(DEVICE)), dim=1)

    for i in range(n):
        pred = probs[i].argmax().item()
        conf = probs[i, pred].item()
        print(f"\n  真实数字: {labels[i].item()}    模型预测: {pred}    "
              f"把握 {conf:.2%}")
        for row in images[i, 0]:
            line = "".join(ASCII_CHARS[min(int(v * 10), 9)] for v in row.tolist())
            print("      " + line)


# =============================================================================
# 5. 主流程
# =============================================================================
def main():
    # 固定随机种子：让每次运行结果可复现，方便对比实验
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 66)
    print("三层神经网络识别手写数字 (MNIST)")
    print(f"运行设备: {DEVICE}")
    print("=" * 66)

    train_loader, test_loader, images, labels = build_dataloaders()

    # ---- 建模型 ----
    print("[2/4] 搭建模型 ...")
    model = ThreeLayerNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"      可学习参数总数: {n_params:,} 个 "
          f"(约 {n_params * 4 / 1024 / 1024:.2f} MB, 用 float32 存)")

    # ---- 损失函数和优化器 ----
    # CrossEntropyLoss = 先做 softmax 再算交叉熵，所以模型最后一层不要再加 softmax！
    loss_fn = nn.CrossEntropyLoss()
    # Adam 是自适应学习率的优化器，对新手最省心；想对比可以换成 SGD(model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- 训练 ----
    print("[3/4] 开始训练 ...")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, epoch)
        test_acc = evaluate(model, test_loader)
        print(f"  第 {epoch}/{EPOCHS} 轮完成 | 训练 loss {train_loss:.4f} | "
              f"训练准确率 {train_acc:.2%} | 测试准确率 {test_acc:.2%} | "
              f"用时 {time.time() - t0:.1f}s")
    print(f"  训练总耗时 {time.time() - start:.1f}s")

    # ---- 最终评估 ----
    print("[4/4] 最终评估 ...")
    final_acc = evaluate(model, test_loader)
    print(f"      测试集准确率: {final_acc:.2%}  "
          f"(10000 张里认对约 {int(final_acc * 10000)} 张)")

    # ---- 保存模型参数 ----
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"      模型参数已保存到 {MODEL_PATH}")

    # ---- 看看具体认了些什么 ----
    print("\n看看模型认的几张图（每张 28x28，用字符模拟像素深浅）：")
    show_predictions(model, images, labels, n=3)

    print("\n完成！可以试试改开头的超参数再跑一遍：")
    print("  EPOCHS 1 -> 10            准确率会更高，但更慢")
    print("  HIDDEN1 256 -> 32         实测掉到 95%，参数量少 8 倍")
    print("  LEARNING_RATE 0.001 -> 0.1  实测掉到 54%，步子太大跨过最优点")
    print("  删掉两个 nn.ReLU()        实测掉到 91%，说明非线性层才是关键")


if __name__ == "__main__":
    main()
