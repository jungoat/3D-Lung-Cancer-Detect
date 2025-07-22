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
from nodul_center_extractor import resize_and_stack_images  # import preprocessing function

# Dice Loss Definition
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

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "seg_model.state"

# Fine-tuning loop for each patient
for pid in range(1, 7):
    print(f"\nProcessing Patient {pid}...")

    # Paths for image, mask, CSV, output
    base_path = str(pid)
    img_base_dir = os.path.join(base_path, "data")
    mask_base_dir = os.path.join(base_path, "masks")
    csv_path = os.path.join(base_path, "nodule_centers.csv")
    save_img_dir = f"resized_imgs_{pid}"
    save_mask_dir = f"resized_masks_{pid}"
    save_model_path = f"finetuned_model_{pid}.state"

    # Step 1: Preprocessing - create resized stacked images and masks
    resize_and_stack_images(csv_path, img_base_dir, mask_base_dir, save_img_dir, save_mask_dir)

    # Step 2: Load dataset and dataloader
    train_dataset = LungStackedDataset(save_img_dir, save_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Step 3: Load pretrained segmentation model
    model = UNetWrapper(in_channels=7, n_classes=1, wf=4, depth=3, batch_norm=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(DEVICE)

    # Step 4: Define loss and optimizer
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 5: Train model
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for imgs, masks, *_ in tqdm(train_loader, desc=f"[Patient {pid}] Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Patient {pid} - Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

    # Step 6: Save fine-tuned model
    torch.save({"model_state": model.state_dict()}, save_model_path)
    print(f"Saved fine-tuned model to {save_model_path}")
