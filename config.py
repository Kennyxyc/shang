import torch
import os

# 训练配置类
class Config:
    # 1. 数据集设置
    data_root = "./data"                # 数据集根目录
    num_classes = 10                 # 你的类别数量，按需修改
    img_size = 224                     # 输入图像尺寸

    # 2. 训练参数
    batch_size = 32
    epochs = 50
    lr = 1e-4                          # 学习率
    weight_decay = 1e-5

    # 3. 加载器设置
    num_workers = 4
    pin_memory = True

    # 4. 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 5. 保存路径
    save_dir = "./checkpoints"
    log_dir = "./logs"
    best_model_path = os.path.join(save_dir, "best.pth")

# 自动创建文件夹
os.makedirs(Config.save_dir, exist_ok=True)
os.makedirs(Config.log_dir, exist_ok=True)