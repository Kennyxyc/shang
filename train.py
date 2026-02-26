import torch
import logging
import time
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from dataset import train_data, train_labels, test_data, test_labels
from config import Config
from model import build_model

# ====================== Logging 配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_acc(outputs, labels):
    predicts = outputs.argmax(dim=1)
    correct = (predicts == labels).sum().item()
    acc = correct / len(labels)
    return acc

# ====================== 使用 v2 的 Dataset ======================
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, labels_np, transform=None):
        self.data = data_np  # [B, H, W, C], uint8, [0,255]
        self.labels = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]  # [H, W, C] numpy
        label = self.labels[index]

        if self.transform is not None:
            # v2 可以直接处理 numpy 数组，无需转 PIL
            img = self.transform(img)

        return img, label

# ====================== Windows 多进程入口 ======================
if __name__ == '__main__':

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0
    patience = 10
    counter = 0

    # ====================== 初始化 ======================
    config = Config()
    epochs = config.epochs
    lr = 1e-3
    batch_size = config.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ====================== 划分验证集 ======================
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # ====================== 使用 v2 重构增强管道 ======================
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # 自动将 [0,255] 转为 [0,1]
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # Dataset & DataLoader
    train_dataset = AugmentedDataset(X_train, y_train, transform=train_transform)
    val_dataset = AugmentedDataset(X_val, y_val, transform=val_transform)
    test_dataset = AugmentedDataset(test_data, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ====================== 训练循环 ======================
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        start_time = time.time()

        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
        for imgs, labels in train_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            acc = compute_acc(outputs, labels)
            train_acc += acc

            # ========== 这里已改：实时显示 loss + acc ==========
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        train_time = time.time() - start_time
        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            # 可选：清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}")
            for imgs, labels in val_bar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                acc = compute_acc(outputs, labels)
                val_acc += acc

                # ========== 这里已加：验证也实时显示 loss + acc ==========
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        # 打印日志
        logger.info(f"Epoch [{epoch+1}/{epochs}] "
                    f"Train loss: {avg_train_loss:.4f} acc: {avg_train_acc:.4f} | "
                    f"Val loss: {avg_val_loss:.4f} acc: {avg_val_acc:.4f} | "
                    f"Time: {train_time:.2f}s")

        # 保存最优模型
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), "checkpoints/best.pth")
            logger.info(f"Save best model with acc: {best_acc:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break