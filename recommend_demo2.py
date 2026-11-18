# -*- coding: utf-8 -*-
"""
recommend_demo.py —— 三层神经网络做电影评分预测（推荐系统）
================================================================================

【任务】
    给"某个用户"和"某部电影"，预测这个用户会给这部电影打几颗星（1~5 星）。
    预测完之后，把分数最高的 10 部没看过的电影推荐给这个用户。

【和 three_layer_nn.py 的关系】
    骨架完全一样：配置 -> 数据 -> 模型 -> 损失/优化器 -> 训练 -> 测试 -> 主程序。
    只换了两处：
      1. 输入从"784 个像素"变成"用户 id + 电影 id"两个整数；
      2. 输出从"10 个类别的打分"变成"1 个预测分数"，任务从分类变成回归。

【数据集】
    MovieLens-100K：943 个用户给 1682 部电影打了 10 万条 1~5 星评分。
    第一次运行自动下载（约 5MB）解压到 ./data/ml-100k/，之后直接读本地。

【运行】
    python recommend_demo.py

【输出】
    每轮打印训练 MSE 和测试 RMSE（单位是"星"），8 轮后 RMSE 约 0.97 星；
    训练完打印某个用户的 Top-10 推荐片单。

【结构】
    1. 基础配置    2. 数据下载与读取   3. 模型定义   4. 损失与优化器
    5. 训练函数    6. 测试函数         7. 推荐函数   8. 主程序入口
"""

import os
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ===================== 1. 基础配置 =====================
# 自动选择设备：有 GPU 用 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 固定随机种子：让每次运行的结果可复现
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 超参数（可自行调整感受效果）
batch_size = 512       # 每批训练的样本数量（评分记录都很小，一批可以多装点）
learning_rate = 0.002  # 学习率
epochs = 8             # 训练总轮数（再多就开始过拟合了）
emb_dim = 32           # 用户/电影各自被表示成多长的向量
hidden_size = 64       # 三层 MLP 第一层的宽度
data_dir = "./data"    # 数据集下载/解压目录
data_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
model_path = "recommend_mlp.pth"  # 训练完把权重存到这里


# ===================== 2. 数据下载与读取 =====================
def download_dataset(data_dir, url):
    """下载并解压数据集，返回解压后的目录；已存在则直接返回。"""
    root_dir = os.path.join(data_dir, "ml-100k")
    if os.path.exists(os.path.join(root_dir, "u.data")):
        return root_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    print(f"下载数据集: {url}")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:  # 网络/证书有问题时，给出手动下载的提示
        raise SystemExit(
            f"下载失败: {exc}\n"
            f"请手动下载 {url}\n"
            f"解压后让这个文件存在: {os.path.abspath(os.path.join(root_dir, 'u.data'))}"
        ) from exc

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return root_dir


def load_ratings(root_dir):
    """读 u.data：每行是 用户id(制表符)电影id(制表符)评分(制表符)时间戳。"""
    users, items, stars = [], [], []
    with open(os.path.join(root_dir, "u.data"), encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            users.append(int(parts[0]) - 1)  # id 从 1 开始，减 1 变成从 0 开始
            items.append(int(parts[1]) - 1)  # 正好可以当 Embedding 的下标
            stars.append(float(parts[2]))

    # user/item 必须是整数（当下标），评分用浮点
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
        torch.tensor(stars, dtype=torch.float32),
    )


def load_titles(root_dir):
    """读 u.item，返回 {电影id: 片名} 的对照表，最后展示推荐结果时要用。"""
    titles = {}
    with open(os.path.join(root_dir, "u.item"), encoding="latin-1") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) >= 2:
                titles[int(parts[0]) - 1] = parts[1]
    return titles


class RatingDataset(Dataset):
    """自定义数据集：实现 __len__ 和 __getitem__ 两个方法就够了。"""

    def __init__(self, users, items, stars):
        self.users = users
        self.items = items
        self.stars = stars

    def __len__(self):
        return len(self.stars)

    def __getitem__(self, index):
        return self.users[index], self.items[index], self.stars[index]


# 下载 + 读取数据
root_dir = download_dataset(data_dir, data_url)
users, items, stars = load_ratings(root_dir)
titles = load_titles(root_dir)
n_users = int(users.max()) + 1
n_items = int(items.max()) + 1
print(f"共 {len(stars)} 条评分 | {n_users} 个用户 | {n_items} 部电影 | 平均 {stars.mean():.2f} 星")

# 打乱后按 8:2 划分训练集和测试集
perm = torch.randperm(len(stars))
n_train = int(len(stars) * 0.8)

train_dataset = RatingDataset(users[perm[:n_train]], items[perm[:n_train]], stars[perm[:n_train]])
test_dataset = RatingDataset(users[perm[n_train:]], items[perm[n_train:]], stars[perm[n_train:]])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ===================== 3. 模型定义 =====================
class MLPRecommender(nn.Module):
    """
    用户 id 查表得到一个向量，电影 id 查表得到一个向量，
    两个向量拼起来，送进三层 MLP，输出一个分数。

    nn.Embedding(n, d) 就是一张 n 行 d 列、可以学习的查找表：
    输入一个整数 id，取出对应的那一行。
    它在推荐系统里就代表"这个用户/这部电影的画像"。
    """

    def __init__(self, n_users, n_items, emb_dim, hidden_dim):
        super().__init__()
        # 两张查找表：用户表、电影表
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        # 三层 MLP：输入 64 维（32+32），中间两层，输出 1 个分数
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, user_ids, item_ids):
        # 查表： (N,) -> (N, 32)
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        # 拼接： (N, 32) + (N, 32) -> (N, 64)
        x = torch.cat([user_vec, item_vec], dim=1)
        # 三层 MLP
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x.squeeze(1)  # (N,1) -> (N,)


# 实例化模型并迁移到对应设备
model = MLPRecommender(n_users, n_items, emb_dim, hidden_size).to(device)


# ===================== 4. 损失函数与优化器 =====================
# 均方误差：回归任务的标准损失函数
criterion = nn.MSELoss()
# Adam 优化器：自适应学习率，收敛稳定
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ===================== 5. 单轮训练函数 =====================
def train_epoch(model, loader, criterion, optimizer, device):
    """把训练集完整跑一遍，返回平均 MSE。"""
    model.train()  # 切换为训练模式
    total_loss = 0.0

    for user_ids, item_ids, ratings in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)

        # ① 前向传播：输出原始分数
        outputs = model(user_ids, item_ids)
        # 把分数压到 0~1，和"评分/5"对齐
        preds = torch.sigmoid(outputs)
        targets = (ratings - 1.0) / 4.0  # 1~5 星 -> 0~1
        # ② 计算损失
        loss = criterion(preds, targets)

        # ③ 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ===================== 6. 测试函数 =====================
def test_epoch(model, loader, device):
    """在测试集上算 RMSE，返回值单位是"星"（越小越好）。"""
    model.eval()  # 切换为评估模式
    squared_error = 0.0
    total = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            outputs = model(user_ids, item_ids)
            preds = torch.sigmoid(outputs) * 4.0 + 1.0  # 0~1 -> 1~5 星

            squared_error += ((preds - ratings) ** 2).sum().item()
            total += ratings.size(0)

    # RMSE = 均方误差开根号
    return (squared_error / total) ** 0.5


# ===================== 7. 推荐函数 =====================
def recommend(model, user_id, seen_items, titles, n_items, k=10):
    """给一个用户推荐 k 部他没看过的电影。"""
    model.eval()
    with torch.no_grad():
        # 这个用户和所有电影配对，一次性算出全部分数
        user_tensor = torch.full((n_items,), user_id, dtype=torch.long, device=device)
        item_tensor = torch.arange(n_items, device=device)
        scores = torch.sigmoid(model(user_tensor, item_tensor)).cpu()

    # 看过的电影分数设为 -1，保证推荐的都是没看过的
    scores[seen_items] = -1.0
    top_items = scores.topk(k).indices.tolist()

    print(f"\n给用户 {user_id} 的 Top-{k} 推荐（他/她已经评过 {len(seen_items)} 部）：")
    for rank, item_id in enumerate(top_items, start=1):
        star = scores[item_id].item() * 4 + 1
        print(f"  {rank:2d}. 预测 {star:.2f} 星 | {titles.get(item_id, '(未知片名)')}")


# ===================== 8. 主程序入口 =====================
if __name__ == "__main__":
    print("开始训练...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_rmse = test_epoch(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | 训练 MSE: {train_loss:.5f} | 测试 RMSE: {test_rmse:.3f} 星")

    # 保存模型权重
    torch.save(model.state_dict(), model_path)
    print(f"\n训练完成！模型权重已保存为 {model_path}")

    # 挑一个评分数量最多的用户，给他生成推荐
    counts = torch.bincount(users, minlength=n_users)
    demo_user = int(counts.argmax().item())
    seen = torch.unique(items[users == demo_user]).tolist()
    recommend(model, demo_user, seen, titles, n_items, k=10)
