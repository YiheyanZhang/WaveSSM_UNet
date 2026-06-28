# 数据加载部分，包含3D地震数据增强
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset


class FaultDataset(Dataset):
    """
    Load_Dataset
    """

    def __init__(self, path=None, mode='train', transform=None, dim=(128, 128, 128),
                 augment=True, file_list=None):
        """
        Args:
            path: 数据根目录，包含 seis/ 与 fault/ 子目录。与 file_list 二选一。
            file_list: [(img_path, label_path), ...] 直接指定文件对，用于子集采样。
        """
        self.path = path
        self.transform = transform
        self.mode = mode
        self.dim = dim
        self.augment = augment
        self.file_list = file_list

        self.image_list, self.label_list = self.load_data()

    def __getitem__(self, index):
        # 使用 np.fromfile 读取二进制文件
        image = np.fromfile(self.image_list[index], dtype=np.float32)
        image = np.reshape(image, self.dim)

        if len(self.label_list) == 0:
            label = np.zeros(image.shape)
        else:
            label = np.fromfile(self.label_list[index], dtype=np.float32)
            label = np.reshape(label, self.dim)

        # 与faultSeg原版一致：z-score标准化
        xm = np.mean(image)
        xs = np.std(image) + 1e-8  # 防止除零
        image = (image - xm) / xs

        # 转置处理
        image = np.transpose(image)
        label = np.transpose(label)

        # 数据增强（仅训练模式）
        if self.mode == 'train' and self.augment:
            image, label = self._augment(image, label)

        img = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        x = torch.from_numpy(np.ascontiguousarray(img))
        y = torch.from_numpy(np.ascontiguousarray(label))

        data = {'x': x.float(), 'y': y.float()}

        return data

    def _augment(self, image, label):
        """
        3D地震数据增强（增强版）

        包含：
        1. 三轴翻转（各50%概率）
        2. 90度旋转（50%概率）
        3. 高斯噪声（30%概率）
        4. 对比度调整（30%概率）
        5. CutOut遮挡 - 已关闭（与小波物理先验冲突）
        """
        # 1. 三轴翻转（增加方向多样性）
        for axis in [0, 1, 2]:
            if random.random() > 0.5:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)

        # 2. 90度旋转（在随机平面上）
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            image = np.rot90(image, k=k, axes=axes)
            label = np.rot90(label, k=k, axes=axes)

        # 3. 高斯噪声（增强鲁棒性）
        if random.random() > 0.7:
            noise_std = random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = image + noise

        # 4. 对比度调整
        if random.random() > 0.7:
            contrast = random.uniform(0.9, 1.1)
            image = image * contrast

        # 5. CutOut遮挡 (已关闭)
        # 关闭原因：CutOut 抹 input 不抹 label，若遮挡压在断层上会要求模型"在零块上预测断层"，
        # 与损失函数矛盾；且零块边界会让小波分解产生人工高频，污染 WaveSSM 的物理先验通路。
        # if random.random() > 0.8:
        #     cut_size = random.randint(8, 20)
        #     d = random.randint(0, image.shape[0] - cut_size)
        #     h = random.randint(0, image.shape[1] - cut_size)
        #     w = random.randint(0, image.shape[2] - cut_size)
        #     image[d:d+cut_size, h:h+cut_size, w:w+cut_size] = 0

        # 确保内存连续
        return image.copy(), label.copy()

    def __len__(self):
        return len(self.image_list)

    def load_data(self):
        """
        :return: (image_list, label_list)
        优先级：file_list > path 扫描
        """
        # 1) 显式 file_list 模式（用于训练集子采样）
        if self.file_list is not None:
            img_list = [p[0] for p in self.file_list]
            if self.mode != 'pred':
                label_list = [p[1] for p in self.file_list]
            else:
                label_list = []
            return img_list, label_list

        # 2) 目录扫描模式（原版逻辑）
        img_list = []
        label_list = []
        label_pred_list = []
        img_path = os.path.join(self.path, 'seis/')
        label_path = os.path.join(self.path, 'fault/')
        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))
            # 由于x和y的文件名一样，所以用一步加载进来
            label_list.append(os.path.join(label_path, item))
        if self.mode != 'pred':
            return img_list, label_list
        else:
            return img_list, label_pred_list