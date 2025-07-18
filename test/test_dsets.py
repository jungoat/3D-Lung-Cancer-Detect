import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SegmentationTestDataset(Dataset):
    def __init__(self, patient_path):
        self.data_path = os.path.join(patient_path, 'data')
        self.mask_path = os.path.join(patient_path, 'masks')

        self.data_files = sorted(os.listdir(self.data_path))
        self.mask_files = sorted(os.listdir(self.mask_path))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.data_path, self.data_files[idx]))
        mask = np.load(os.path.join(self.mask_path, self.mask_files[idx]))

        # 🔄 반시계 방향 90도 회전 + 복사로 stride 문제 해결
        img = np.rot90(img, k=1).copy()
        mask = np.rot90(mask, k=1).copy()

        # Tensor 변환
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor


class SegmentationTestStackedDataset(Dataset):
    def __init__(self, patient_path, window=3):
        self.data_path = os.path.join(patient_path, 'data')
        self.mask_path = os.path.join(patient_path, 'masks')
        self.window = window  # 중심 기준 ±window → 총 2*window+1

        self.data_files = sorted(os.listdir(self.data_path))
        self.mask_files = sorted(os.listdir(self.mask_path))
        self.len = len(self.data_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        indices = list(range(idx - self.window, idx + self.window + 1))
        indices = [min(max(i, 0), self.len - 1) for i in indices]  # 경계 보정

        # 7슬라이스 stack
        slices = [np.load(os.path.join(self.data_path, self.data_files[i])) for i in indices]
        slices = [np.rot90(s, k=1).copy() for s in slices]  # 회전 + stride 문제 해결
        img_stack = np.stack(slices)  # [7, 256, 256]

        # GT 마스크는 중심 슬라이스 하나만 사용
        mask = np.load(os.path.join(self.mask_path, self.mask_files[idx]))
        mask = np.rot90(mask, k=1).copy()

        img_tensor = torch.tensor(img_stack, dtype=torch.float32)       # [7, 256, 256]
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]

        return img_tensor, mask_tensor