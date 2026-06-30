import os
import sys
import time

# Ensure root workspace is in system path for clean execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import cv2
from src.generator import generate_synthetic_slice
from src.segmentation import UNet, compute_dice, compute_iou
from src.classification import get_classification_model
from src.mlflow_tracker import log_mlflow_run

# =====================================================================
# 1. Dataset Class (Synthetic Fallback & Custom Local Folders)
# =====================================================================

class MRIDataset(Dataset):
    def __init__(self, mode="train", length=100, local_dir=None, seed=42):
        """
        PyTorch Dataset for MRI images.
        If local_dir is provided and contains 'images' and 'masks', loads from it.
        Otherwise, generates high-fidelity synthetic MRI slices on the fly.
        """
        self.local_dir = local_dir
        self.mode = mode
        self.length = length
        self.seed = seed
        
        if local_dir and os.path.exists(os.path.join(local_dir, "images")):
            self.image_dir = os.path.join(local_dir, "images")
            self.mask_dir = os.path.join(local_dir, "masks")
            self.file_names = sorted(os.listdir(self.image_dir))
            self.use_synthetic = False
            self.length = len(self.file_names)
        else:
            self.use_synthetic = True
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if not self.use_synthetic:
            # Load real image/mask from disk
            img_path = os.path.join(self.image_dir, self.file_names[idx])
            # The mask has the same basename but .png extension
            base_name, _ = os.path.splitext(self.file_names[idx])
            mask_path = os.path.join(self.mask_dir, base_name + ".png")
            
            # Read grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            
            # Extract label from filename
            if "_gl_" in base_name:
                class_label = 1 # Glioma
            elif "_me_" in base_name:
                class_label = 2 # Meningioma
            elif "_pi_" in base_name:
                class_label = 3 # Pituitary
            elif "_nt_" in base_name:
                class_label = 0 # No tumor
            else:
                class_label = 0
        else:
            # Generate synthetic image on the fly
            # Use deterministic seed per index to keep dataset static during epochs
            seed_val = self.seed + idx if self.mode == "train" else self.seed + 1000 + idx
            np.random.seed(seed_val)
            
            modality = np.random.choice(["FLAIR", "T2", "T1"])
            tumor_present = np.random.rand() > 0.15 # 85% probability of tumor
            tumor_size = np.random.uniform(10, 40) if tumor_present else 0
            t_loc = (np.random.uniform(-0.35, 0.35), np.random.uniform(-0.35, 0.35))
            noise = np.random.uniform(0.01, 0.05)
            
            img, mask = generate_synthetic_slice(
                modality=modality,
                tumor_present=tumor_present,
                tumor_size=tumor_size,
                tumor_loc=t_loc,
                noise_level=noise,
                seed=seed_val
            )
            
            if tumor_present:
                class_label = np.random.choice([1, 2, 3])
            else:
                class_label = 0
            
        # Resize to 224x224 (Standard classifier / segmenter size)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Convert to PyTorch tensors (Channel, Height, Width)
        img_t = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0)
        
        return img_t, mask_t, torch.tensor(class_label, dtype=torch.long)


# =====================================================================
# 2. Training Loops
# =====================================================================

def train_segmentation(epochs=3, batch_size=8, lr=1e-3, device="cpu"):
    """
    Trains U-Net segmentation model.
    """
    print("\n--- Initializing Segmentation Model Training ---")
    if os.path.exists("data/train"):
        dataset_train = MRIDataset(mode="train", local_dir="data/train")
        dataset_val = MRIDataset(mode="val", local_dir="data/test")
        print(f"[INFO] Loaded local dataset: {len(dataset_train)} training samples, {len(dataset_val)} testing/validation samples (70/30 split).")
    else:
        dataset_train = MRIDataset(mode="train", length=64, seed=42)
        dataset_val = MRIDataset(mode="val", length=16, seed=1337)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Combined Loss: BCE + Dice Loss
    def bce_dice_loss(pred, target):
        bce = nn.BCELoss()(pred, target)
        
        # Dice loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1e-5) / (pred_flat.sum() + target_flat.sum() + 1e-5)
        return bce + (1.0 - dice)
        
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in loader_train:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = bce_dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(loader_train.dataset)
        
        # Validation
        model.eval()
        val_dice = []
        val_iou = []
        with torch.no_grad():
            for imgs, masks, _ in loader_val:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                
                # Compute metrics
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                for p, m in zip(preds_np, masks_np):
                    val_dice.append(compute_dice(p[0], m[0]))
                    val_iou.append(compute_iou(p[0], m[0]))
                    
        mean_dice = np.mean(val_dice)
        mean_iou = np.mean(val_iou)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}")
        
    # Save model weights
    save_path = "./unet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Segmentation weights saved to {save_path}")
    
    # Log run details to MLflow / Local DB
    elapsed = time.time() - start_time
    log_mlflow_run(
        run_name=f"UNet_Train_{int(time.time())}",
        params={"epochs": epochs, "lr": lr, "batch_size": batch_size, "device": device},
        metrics={"final_val_dice": float(mean_dice), "final_val_iou": float(mean_iou), "training_duration_sec": elapsed},
        artifacts={"unet_weights": save_path}
    )
    return model

def train_classification(epochs=3, batch_size=8, lr=1e-3, device="cpu"):
    """
    Trains ResNet50/LightweightCNN classification model.
    """
    print("\n--- Initializing Classification Model Training ---")
    if os.path.exists("data/train"):
        dataset_train = MRIDataset(mode="train", local_dir="data/train")
        dataset_val = MRIDataset(mode="val", local_dir="data/test")
        print(f"[INFO] Loaded local dataset: {len(dataset_train)} training samples, {len(dataset_val)} testing/validation samples (70/30 split).")
    else:
        dataset_train = MRIDataset(mode="train", length=64, seed=42)
        dataset_val = MRIDataset(mode="val", length=16, seed=1337)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    model = get_classification_model("ResNet50", num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, _, labels in loader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(loader_train.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, _, labels in loader_val:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    # Save model weights
    save_path = "./classifier_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Classification weights saved to {save_path}")
    
    # Log run details to MLflow / Local DB
    elapsed = time.time() - start_time
    log_mlflow_run(
        run_name=f"ResNet50_Train_{int(time.time())}",
        params={"epochs": epochs, "lr": lr, "batch_size": batch_size, "device": device},
        metrics={"final_val_acc": float(val_acc), "training_duration_sec": elapsed},
        artifacts={"classifier_weights": save_path}
    )
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing training on: {device.upper()}")
    
    # Train both models
    train_segmentation(epochs=2, batch_size=8, device=device)
    train_classification(epochs=2, batch_size=8, device=device)
    print("\n--- Training Pipeline Executed Successfully ---")
