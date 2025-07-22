import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from skimage import measure


def visualize_prediction(img_tensor, mask_tensor, pred_array, filename, channel=3):
    # Unbatch
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor[0]

    # Convert tensors to numpy arrays
    img_np = img_tensor[channel].cpu().numpy()
    mask_np = mask_tensor[0].cpu().numpy()

    plt.figure(figsize=(12, 4))

    # Input slice
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title(f"Input Slice (channel {channel})\n{filename}")

    # GT mask
    contours = measure.find_contours(mask_np, level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='blue')
    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Ground Truth Mask")

    # Prediction (rescaled for visibility)
    plt.subplot(1, 3, 3)
    plt.imshow(pred_array * 255, cmap='gray', vmin=0, vmax=255)
    pred_contours = measure.find_contours(pred_array, level=0.5)
    for contour in pred_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
    plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()
