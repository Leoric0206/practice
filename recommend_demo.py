# -*- coding: utf-8 -*-
"""
recommend_demo.py —— 把"三层神经网络"用到推荐系统上（进阶样例，选做）
================================================================================

上一个程序 three_layer_nn.py 是"看图分类"，这个程序是"猜用户会给电影打几分"。
两个程序的骨架一模一样：

    准备数据 -> 搭 nn.Module -> 选损失函数和优化器 -> 训练循环 -> 评估

区别只在两处：
    1. 输入不再是 784 个像素，而是"用户 id + 电影 id"这两个整数；
    2. 输出不再是 10 个类别的打分，而是 1 个预测分数（1~5 星）。

【数据集】MovieLens-100K：943 个用户给 1682 部电影打了 10 万条 1~5 星评分。
    第一次运行会自动下载（约 5MB）解压到 ./data/ml-100k/，之后直接读本地。

【怎么运行】
    python recommend_demo.py

【这个模型在推荐系统里叫什么】
    这就是 "Neural Collaborative Filtering"(NCF) 里 MLP 分支的简化版，
    也叫 GMF/MLP 双塔模型的一半：用户塔 + 物品塔 -> 拼接 -> 三层 MLP -> 分数。
    矩阵分解(Matrix Factorization)是它的"去激活函数"版本。
"""

import os
import random
import urllib.request
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# 0. 超参数
# =============================================================================
DATA_DIR = "data"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
EMB_DIM = 32           # 用户/电影各自被表示成多长的向量
HIDDEN = 64            # MLP 第一层宽度
BATCH_SIZE = 512
EPOCHS = 8             # 8 轮左右测试误差最低，再多就开始过拟合了
LEARNING_RATE = 0.002
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. 下载并读取数据
# =============================================================================
def ensure_ml100k(data_dir: str) -> str:
    """保证数据在本地，返回 ml-100k 目录路径。"""
    root = os.path.join(data_dir, "ml-100k")
    if os.path.exists(os.path.join(root, "u.data")):
        return root

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    print(f"  下载 {ML100K_URL}")
    try:
        urllib.request.urlretrieve(ML100K_URL, zip_path)
    except Exception as exc:  # 网络/证书有问题时，给出能自己动手的提示
        raise SystemExit(
            f"  下载失败：{exc}\n"
            f"  可以手动下载 {ML100K_URL}\n"
            f"  解压后让这个文件存在：{os.path.abspath(os.path.join(root, 'u.data'))}\n"
            f"  然后再运行本程序。"
        ) from exc

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return root


def load_ratings(root: str):
    """读 u.data：每行是 用户id \\t 电影id \\t 评分 \\t 时间戳。"""
    users, items, stars = [], [], []
    with open(os.path.join(root, "u.data"), encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            users.append(int(parts[0]) - 1)     # id 从 1 开始，减 1 变成从 0 开始
            items.append(int(parts[1]) - 1)     # 正好可以当 Embedding 的下标用
            stars.append(float(parts[2]))
    # 转成张量：user/item 必须整数（当下标用），评分要浮点
    return (torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(stars, dtype=torch.float32))


def load_titles(root: str) -> dict:
    """读 u.item，拿"电影 id -> 片名"的对照表，最后展示推荐结果时用。"""
    titles = {}
    with open(os.path.join(root, "u.item"), encoding="latin-1") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) >= 2:
                titles[int(parts[0]) - 1] = parts[1]
    return titles


class RatingDataset(Dataset):
    """
    自定义数据集就实现两个魔术方法：
        __len__      一共有多少条样本
        __getitem__  取第 k 条样本，返回一个元组（和 C 里返回 struct 差不多）
    DataLoader 会自动帮你做"取 batch、打乱、拼成张量"这些活。
    """

    def __init__(self, users, items, stars):
        self.users = users
        self.items = items
        self.stars = stars

    def __len__(self):
        return len(self.stars)

    def __getitem__(self, k):
        return self.users[k], self.items[k], self.stars[k]


# =============================================================================
# 2. 模型：Embedding + 三层 MLP
# =============================================================================
class MLPRecommender(nn.Module):
    """
    nn.Embedding(数量, 维度) 本质就是一张可以学习的查找表：
        输入一个整数 id，输出一行长度 EMB_DIM 的向量。
    它和 nn.Linear 的区别：Linear 是"乘矩阵"，Embedding 是"按行取"，
    但取出来的那一行同样是可学习参数，同样有梯度，一样会被优化器更新。

    用户向量(32) 和 电影向量(32) 拼成 64 维，送进三层 MLP，输出一个分数。
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, EMB_DIM)
        self.item_emb = nn.Embedding(n_items, EMB_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM * 2, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1),          # 回归任务：输出 1 个数字
        )

    def forward(self, user_ids, item_ids):
        # 形状变化： (N,) -> (N, 32)；两个拼起来 -> (N, 64)
        x = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids)], dim=1)
        return self.mlp(x).squeeze(1)           # (N, 1) -> (N,)


# =============================================================================
# 3. 训练和评估
# =============================================================================
def rmse_in_stars(pred01: torch.Tensor, stars: torch.Tensor) -> float:
    """预测值在 0~1 之间，先乘回 4 加 1 变回 1~5 星，再算均方根误差。"""
    pred = pred01 * 4.0 + 1.0
    return torch.sqrt(torch.mean((pred - stars) ** 2)).item()


def run_epoch(model, loader, loss_fn, optimizer=None):
    """optimizer 传了就是训练，不传就是评估。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    preds = []
    targets = []
    for users, items, stars in loader:
        users, items = users.to(DEVICE), items.to(DEVICE)
        stars = stars.to(DEVICE)

        logits = model(users, items)
        pred01 = torch.sigmoid(logits)          # 压到 0~1，和"评分/5"对齐
        loss = loss_fn(pred01, (stars - 1.0) / 4.0)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds.append(pred01.detach().cpu())
        targets.append(stars.detach().cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return total_loss / len(loader), rmse_in_stars(preds, targets)


# =============================================================================
# 4. 用训好的模型给某个用户推荐电影
# =============================================================================
def recommend_for_user(model, user_id, seen_items, titles, n_items, k=10):
    model.eval()
    with torch.no_grad():
        # 一次性给这个用户算一遍所有电影的分数（1682 部，很少，直接全算）
        users = torch.full((n_items,), user_id, dtype=torch.long, device=DEVICE)
        items = torch.arange(n_items, device=DEVICE)
        scores = torch.sigmoid(model(users, items)).cpu()

    # 把用户已经看过的电影分数设为 -1，保证推荐的都是没看过的
    scores[seen_items] = -1.0
    top_items = scores.topk(k).indices.tolist()

    print(f"  给用户 {user_id} 的 Top-{k} 推荐（他/她已经评过 {len(seen_items)} 部）：")
    for rank, item in enumerate(top_items, start=1):
        print(f"    {rank:2d}. 预测 {scores[item].item() * 4 + 1:.2f} 星 | "
              f"{titles.get(item, '(未知片名)')}")


# =============================================================================
# 5. 主流程
# =============================================================================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 66)
    print("MovieLens-100K 电影评分预测（Embedding + 三层 MLP）")
    print(f"运行设备: {DEVICE}")
    print("=" * 66)

    # ---- 数据 ----
    print("[1/4] 准备数据 ...")
    root = ensure_ml100k(DATA_DIR)
    users, items, stars = load_ratings(root)
    titles = load_titles(root)
    n_users = int(users.max()) + 1
    n_items = int(items.max()) + 1
    print(f"      {len(stars)} 条评分 | {n_users} 个用户 | {n_items} 部电影 | "
          f"平均 {stars.mean():.2f} 星")

    # 80% 训练 / 20% 测试（随机打乱后再切，避免"某个用户全在训练集"的偏差）
    n = len(stars)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    n_train = int(n * 0.8)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(
        RatingDataset(users[train_idx], items[train_idx], stars[train_idx]),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        RatingDataset(users[test_idx], items[test_idx], stars[test_idx]),
        batch_size=BATCH_SIZE, shuffle=False)
    print(f"      训练集 {len(train_idx)} 条，测试集 {len(test_idx)} 条")

    # ---- 模型 ----
    print("[2/4] 搭建模型 ...")
    model = MLPRecommender(n_users, n_items).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      可学习参数: {n_params:,} 个 "
          f"(其中用户表 {n_users * EMB_DIM:,} + 电影表 {n_items * EMB_DIM:,})")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- 训练 ----
    print("[3/4] 开始训练 ...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_rmse = run_epoch(model, train_loader, loss_fn, optimizer)
        _, test_rmse = run_epoch(model, test_loader, loss_fn)
        print(f"  第 {epoch:2d}/{EPOCHS} 轮 | 训练 MSE {train_loss:.5f} | "
              f"训练 RMSE {train_rmse:.3f} 星 | 测试 RMSE {test_rmse:.3f} 星")

    # ---- 推荐 ----
    print("[4/4] 生成推荐 ...")
    # 挑一个评分数量最多的用户，当作演示对象（评得多，兴趣信息才够）
    counts = torch.bincount(users, minlength=n_users)
    demo_user = int(counts.argmax().item())
    # 这个用户评过哪些电影 -> 推荐时要排除掉
    seen = torch.unique(items[users == demo_user]).tolist()
    recommend_for_user(model, demo_user, seen, titles, n_items, k=10)

    print("\n说明：测试 RMSE 0.97 星左右，意思是「平均猜错 1 颗星」。")
    print("      想亲眼看过拟合：把 EPOCHS 改成 30，你会看到训练 RMSE 一直降、")
    print("      测试 RMSE 反而升上去。可以试试 EMB_DIM 改成 8 或 64，")
    print("      或者去掉 MLP 只做向量内积（那就退化成矩阵分解）。")


if __name__ == "__main__":
    main()
