import os
import sys

sys.path.append("c:/Users/ADMIN/Desktop/lung")

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from p2ch13.model import UNetWrapper
from test_dsets import LungStackedDataset
from nodul_center_extractor import resize_and_stack_images  # Preprocessing function

# Dice Loss definition
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
CHECKPOINT_PATH = "seg_model.state"
SAVE_MODEL_PATH = "finetuned_all_patients.state"

# Unified save dirs
save_img_dir = "resized_imgs_all"
save_mask_dir = "resized_masks_all"
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)

# Step 1: Preprocess all 6 patients and save into one folder
for pid in range(1, 7):
    print(f"Processing patient {pid}...")
    base_path = str(pid)
    img_base_dir = os.path.join(base_path, "data")
    mask_base_dir = os.path.join(base_path, "masks")
    csv_path = os.path.join(base_path, "nodule_centers.csv")

    # Save results with filename prefix
    resize_and_stack_images(
        csv_path,
        img_base_dir,
        mask_base_dir,
        save_img_dir,
        save_mask_dir,
        target_size=64
    )

# Step 2: Load combined dataset
dataset = LungStackedDataset(save_img_dir, save_mask_dir)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 3: Load model
model = UNetWrapper(in_channels=7, n_classes=1, wf=4, depth=3, batch_norm=True)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"], strict=True)
model = model.to(DEVICE)

# Step 4: Loss and optimizer
criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 5: Train
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for imgs, masks, *_ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")

# Step 6: Save model
torch.save({"model_state": model.state_dict()}, SAVE_MODEL_PATH)
print(f"Model saved to {SAVE_MODEL_PATH}")
