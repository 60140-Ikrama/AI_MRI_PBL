import os
import sys
import numpy as np
import cv2

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import generate_synthetic_slice

def create_dataset_split(split_name, num_samples, start_id, seed):
    np.random.seed(seed)
    
    images_dir = f"data/{split_name}/images"
    masks_dir = f"data/{split_name}/masks"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"[INFO] Generating {num_samples} samples for {split_name} split...")
    
    for i in range(num_samples):
        patient_id = start_id + i
        modality = np.random.choice(["FLAIR", "T2", "T1"])
        tumor_present = np.random.rand() > 0.15 # 85% tumor rate
        tumor_size = np.random.uniform(12, 38) if tumor_present else 0.0
        t_loc = (np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
        noise = np.random.uniform(0.01, 0.04)
        
        img, mask = generate_synthetic_slice(
            modality=modality,
            tumor_present=tumor_present,
            tumor_size=tumor_size,
            tumor_loc=t_loc,
            noise_level=noise,
            seed=seed + i
        )
        
        # Save as uint8 PNG
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        
        filename = f"scan_{patient_id:04d}.png"
        cv2.imwrite(os.path.join(images_dir, filename), img_u8)
        cv2.imwrite(os.path.join(masks_dir, filename), mask_u8)

def main():
    print("==================================================")
    print("[INFO] Creating Local Dataset (70% Train, 30% Test)...")
    
    # 70% Train (70 samples), 30% Test (30 samples)
    create_dataset_split("train", 70, start_id=1, seed=42)
    create_dataset_split("test", 30, start_id=71, seed=1337)
    
    print("[SUCCESS] Dataset successfully generated and split (70/30).")
    print("  Train location: data/train/")
    print("  Test location:  data/test/")
    print("==================================================")

if __name__ == "__main__":
    main()
