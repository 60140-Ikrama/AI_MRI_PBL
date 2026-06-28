import os
import sys
import time
import json
import numpy as np
import torch
import cv2
import pandas as pd

# Ensure workspace root is in system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import generate_synthetic_slice
from src.preprocessing import run_preprocessing_pipeline
from src.segmentation import (
    UNet, AttentionUNet, UNetPlusPlus, MaskRCNNWrapper,
    get_segmentation_metrics
)

def main():
    print("==================================================")
    print("[INFO] Generating cohort of 30 synthetic patients for segmentation...")
    cohort = []
    np.random.seed(42)
    for i in range(30):
        modality = np.random.choice(["FLAIR", "T2", "T1"])
        # Always tumor present for segmentation evaluation
        tumor_present = True
        tumor_size = np.random.uniform(15, 35)
        t_loc = (np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25))
        noise = np.random.uniform(0.02, 0.06)
        
        img, mask = generate_synthetic_slice(
            modality=modality,
            tumor_present=tumor_present,
            tumor_size=tumor_size,
            tumor_loc=t_loc,
            noise_level=noise,
            seed=i
        )
        
        prep_img, _ = run_preprocessing_pipeline(img, ['strip', 'noise', 'clahe', 'norm'])
        
        cohort.append({
            "id": f"PX_{100+i}",
            "image": prep_img,
            "mask": mask
        })
    print(f"[INFO] Generated {len(cohort)} patient scans successfully.")
    
    # 2. Setup models
    device = "cpu"
    unet = UNet(in_channels=1, out_channels=1).to(device)
    att_unet = AttentionUNet(in_channels=1, out_channels=1).to(device)
    unetpp = UNetPlusPlus(in_channels=1, out_channels=1).to(device)
    maskrcnn = MaskRCNNWrapper(in_channels=1).to(device)
    
    # Load pre-trained U-Net weights if they exist
    unet_weights_path = "./unet_model.pth"
    if os.path.exists(unet_weights_path):
        try:
            unet.load_state_dict(torch.load(unet_weights_path, map_location=device))
            print("[INFO] Loaded pre-trained U-Net weights.")
        except Exception as e:
            print(f"[WARNING] Failed to load U-Net weights: {e}")
            
    models = {
        "U-Net": unet,
        "Attention U-Net": att_unet,
        "U-Net++": unetpp,
        "Mask R-CNN": maskrcnn
    }
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.eval()
        
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        hausdorff_scores = []
        bf1_scores = []
        latencies = []
        
        for p in cohort:
            img_t = torch.tensor(p["image"], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            gt = p["mask"]
            
            t0 = time.perf_counter()
            with torch.no_grad():
                if name == "Mask R-CNN":
                    # Mask R-CNN wrapper returns (mask, bbox)
                    pred_tensor, _ = model(img_t)
                else:
                    pred_tensor = model(img_t)
                    
            latencies.append(time.perf_counter() - t0)
            
            pred = pred_tensor.squeeze().cpu().numpy()
            
            # Crop predicted mask to same shape
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                
            metrics = get_segmentation_metrics(pred, gt)
            
            dice_scores.append(metrics["Dice"])
            iou_scores.append(metrics["IoU"])
            precision_scores.append(metrics["Precision"])
            recall_scores.append(metrics["Recall"])
            hausdorff_scores.append(metrics["Hausdorff"])
            bf1_scores.append(metrics["Boundary F1"])
            
        results[name] = {
            "Dice": float(np.mean(dice_scores)),
            "IoU": float(np.mean(iou_scores)),
            "Precision": float(np.mean(precision_scores)),
            "Recall": float(np.mean(recall_scores)),
            "Hausdorff": float(np.mean(hausdorff_scores)),
            "Boundary_F1": float(np.mean(bf1_scores)),
            "Latency_ms": float(np.mean(latencies)) * 1000
        }
        
    # Save results to JSON
    with open("segmentation_benchmarks.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n================== SEGMENTATION BENCHMARK SUMMARY (Q2) ==================")
    df = pd.DataFrame(results).T
    print(df.to_string())
    print("=========================================================================")

if __name__ == "__main__":
    main()
