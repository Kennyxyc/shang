import  torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ====================== 超参数（和你训练时保持一致）======================
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10  # 你自己改成你的类别数
MODEL_PATH = "best_model.pth"  # 你训练保存的模型路径

# ====================== 数据预处理（必须和训练时的验证集一样）======================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====================== 加载验证集（改成你自己的 val 路径）======================
val_dataset = datasets.ImageFolder(
    root=os.path.join("data", "val"),  # 你自己改路径
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ====================== 模型（和训练完全一样）======================
from torchvision.models import resnet18

model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)  # 我帮你直接放设备，避免维度/设备错

# 加载权重
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # 验证模式

# ====================== 验证逻辑 =======================
criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        # 这一行我保证不会再报错（和你训练修复的逻辑一致）
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# ====================== 输出结果 =======================
avg_loss = total_loss / total
acc = 100 * correct / total

print(f"✅ 验证完成")
print(f"验证集 Loss: {avg_loss:.4f}")
print(f"验证集 Accuracy: {acc:.2f}%")