import os
import numpy as np
import csv
import cv2
import torch

from torch.utils.data import Dataset
from scipy.ndimage import center_of_mass, label


def adjust_brightness_contrast(img, brightness=0.9, contrast=0.5):
    img = img + brightness
    img = (img - 0.5) * (1 + contrast) + 0.5
    img = np.clip(img, 0, 1)
    return img


class LungStackedDataset(Dataset):
    def __init__(self, img_dir, mask_dir, hu_min=-1000, hu_max=400):
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = np.load(img_path)
        mask = np.load(mask_path)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img, self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)
        img = img.astype(np.float32)

        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.clip(mask, 0, 1)
        mask = (mask > 0.5).astype(np.float32)

        img_t = torch.from_numpy(img).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return img_t, mask_t, self.img_files[idx]


def extract_nodule_centers(mask_dir, csv_path):

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slice_name', 'center_row', 'center_col'])

        mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.endswith('.npy')],
            key=lambda x: int(x.replace('.npy', ''))
        )
        
        for mask_file in mask_files:
            mask = np.load(os.path.join(mask_dir, mask_file))
            if np.any(mask == 1):
                labeled_mask, num_features = label(mask)
                for lbl in range(1, num_features + 1):
                    region = (labeled_mask == lbl)
                    center = center_of_mass(region)
                    center_row, center_col = map(int, center)
                    writer.writerow([mask_file, center_row, center_col])

## resize
def get_slice_index(slice_name):
    return int(os.path.splitext(slice_name)[0])


def resize_2d(img_2d, target_size=64, is_mask=False):
    resized = cv2.resize(img_2d, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    if is_mask:
        return np.rot90(resized, k=1).astype(np.float32)

    img_hu = resized * 1400 - 1000                
    img_hu = np.clip(img_hu, -1000, 400)              
    img_norm = (img_hu + 1000) / 1400 * 2 - 1    
    return np.rot90(img_norm, k=1).astype(np.float32)


def resize_and_stack_images(csv_path, img_base_dir, mask_base_dir, save_img_dir, save_mask_dir, stack_depth=7, target_size=64):
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    half_stack = stack_depth // 2

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader) 

        for slice_name, r, c in reader:
            center_idx = get_slice_index(slice_name)
            slice_indices = [center_idx + offset for offset in range(-half_stack, half_stack + 1)]
            img_stack = []

            for idx in slice_indices:
                filename = f"{idx}.npy"
                img_path = os.path.join(img_base_dir, filename)

                if not os.path.exists(img_path):
                    img_slice = np.zeros((256, 256), dtype=np.float32)
                else:
                    img_slice = np.load(img_path)
                    if img_slice.ndim == 3:
                        img_slice = img_slice[0]
                    elif img_slice.ndim != 2:
                        img_slice = np.zeros((256, 256), dtype=np.float32)
                    img_slice = img_slice.astype(np.float32)
                    if img_slice.size == 0 or not np.isfinite(img_slice).all():
                        img_slice = np.zeros((256, 256), dtype=np.float32)

                resized = resize_2d(img_slice, target_size)
                img_stack.append(resized)

            img_stack_np = np.stack(img_stack, axis=0)  # (stack_depth, H, W)

            mask_path = os.path.join(mask_base_dir, slice_name)
            if not os.path.exists(mask_path):
                mask = np.zeros((256, 256), dtype=np.float32)
            else:
                mask = np.load(mask_path).astype(np.float32)
                if mask.ndim != 2 or not np.isfinite(mask).all():
                    mask = np.zeros((256, 256), dtype=np.float32)
            resized_mask = resize_2d(mask, target_size, is_mask=True)

            np.save(os.path.join(save_img_dir, slice_name), img_stack_np)
            np.save(os.path.join(save_mask_dir, slice_name), resized_mask)



def resize_and_stack_images_brightness(csv_path, img_base_dir, mask_base_dir, save_img_dir, save_mask_dir, stack_depth=7, target_size=64):
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    half_stack = stack_depth // 2

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)

        for slice_name, r, c in reader:
            center_idx = get_slice_index(slice_name)
            slice_indices = [center_idx + offset for offset in range(-half_stack, half_stack + 1)]
            img_stack = []

            for idx in slice_indices:
                filename = f"{idx}.npy"
                img_path = os.path.join(img_base_dir, filename)

                if not os.path.exists(img_path):
                    img_slice = np.zeros((256, 256), dtype=np.float32)
                else:
                    img_slice = np.load(img_path)
                    if img_slice.ndim == 3:
                        img_slice = img_slice[0]
                    elif img_slice.ndim != 2:
                        img_slice = np.zeros((256, 256), dtype=np.float32)
                    if img_slice.dtype != np.float32:
                        img_slice = img_slice.astype(np.float32)
                    if img_slice.size == 0 or not np.isfinite(img_slice).all():
                        img_slice = np.zeros((256, 256), dtype=np.float32)


                img_slice = adjust_brightness_contrast(img_slice, brightness=0.1, contrast=0.2)

                resized = resize_2d(img_slice, target_size)
                img_stack.append(resized)

            img_stack_np = np.stack(img_stack, axis=0) 

            mask_path = os.path.join(mask_base_dir, slice_name)
            if not os.path.exists(mask_path):
                mask = np.zeros((256, 256), dtype=np.float32)
            else:
                mask = np.load(mask_path).astype(np.float32)
                if mask.ndim != 2 or not np.isfinite(mask).all():
                    mask = np.zeros((256, 256), dtype=np.float32)
            resized_mask = resize_2d(mask, target_size)

            np.save(os.path.join(save_img_dir, slice_name), img_stack_np)
            np.save(os.path.join(save_mask_dir, slice_name), resized_mask)







## crop
def crop_center_2d(img_2d, center, crop_size=64):
    r, c = center
    h, w = img_2d.shape

    half = crop_size // 2
    r_start = max(r - half, 0)
    r_end = min(r + half, h)
    c_start = max(c - half, 0)
    c_end = min(c + half, w)

    cropped = img_2d[r_start:r_end, c_start:c_end]

    pad_r = crop_size - cropped.shape[0]
    pad_c = crop_size - cropped.shape[1]
    if pad_r > 0 or pad_c > 0:
        cropped = np.pad(cropped,
                         ((0, pad_r), (0, pad_c)),
                         mode='constant',
                         constant_values=0)
    return cropped


def get_slice_index(slice_name):
    return int(os.path.splitext(slice_name)[0])


def crop_and_stack(csv_path, img_base_dir, mask_base_dir, save_img_dir, save_mask_dir, stack_depth=7, crop_size=64):
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    half_stack = stack_depth // 2

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵

        for slice_name, r, c in reader:
            r, c = int(r), int(c)
            center_idx = get_slice_index(slice_name)
            slice_indices = [center_idx + offset for offset in range(-half_stack, half_stack + 1)]

            img_stack = []
            for idx in slice_indices:
                filename = f"{idx}.npy"
                img_path = os.path.join(img_base_dir, filename)

                if not os.path.exists(img_path):
                    img_slice = np.zeros((256, 256), dtype=np.float32)
                else:
                    img_slice = np.load(img_path)
                    if img_slice.ndim == 3:
                        img_slice = img_slice[0]  # 7채널 중 0번 슬라이스 사용 시
                    elif img_slice.ndim == 2:
                        pass
                    else:
                        raise ValueError(f"Unexpected img_slice ndim: {img_slice.ndim}")

                cropped = crop_center_2d(img_slice, (r, c), crop_size)
                img_stack.append(cropped)

            img_stack_np = np.stack(img_stack, axis=0)  # (stack_depth, crop_size, crop_size)

            mask_path = os.path.join(mask_base_dir, slice_name)
            mask = np.load(mask_path)
            cropped_mask = crop_center_2d(mask, (r, c), crop_size)

            np.save(os.path.join(save_img_dir, slice_name), img_stack_np)
            np.save(os.path.join(save_mask_dir, slice_name), cropped_mask)
