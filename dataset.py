import pickle
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# 数据集所在的根目录，请根据你的实际情况修改
DATA_DIR = r"C:\Users\23801\PycharmProjects\PythonProject5\data"


def unpickle(file_name):
    """从CIFAR-10的批处理文件中加载数据"""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径是否正确。")
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_data():
    """加载并预处理所有训练和测试数据"""
    # 加载训练数据
    train_data_list = []
    train_labels_list = []
    for i in range(1, 6):
        batch = unpickle(f'data_batch_{i}')
        train_data_list.append(batch[b'data'])
        train_labels_list.extend(batch[b'labels'])

    # 转换为numpy数组并归一化
    train_data = np.concatenate(train_data_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_data = train_data.astype('float32') / 255.0
    train_labels = np.array(train_labels_list)

    # 加载测试数据
    test_batch = unpickle('test_batch')
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.astype('float32') / 255.0
    test_labels = np.array(test_batch[b'labels'])

    # 转换为one-hot编码
    train_labels_onehot = to_categorical(train_labels, 10)
    test_labels_onehot = to_categorical(test_labels, 10)

    return train_data, train_labels, test_data, test_labels, train_labels_onehot, test_labels_onehot


# 在模块被导入时自动加载数据
train_data, train_labels, test_data, test_labels, train_labels_onehot, test_labels_onehot = load_data()

print("数据集加载并预处理完成！")
print(f"训练集形状: {train_data.shape}, 标签形状: {train_labels_onehot.shape}")
print(f"测试集形状: {test_data.shape}, 标签形状: {test_labels_onehot.shape}")