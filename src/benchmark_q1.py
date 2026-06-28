import os
import sys
import time
import json
import numpy as np
import torch
import pandas as pd

# Ensure workspace root is in system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import generate_synthetic_slice
from src.preprocessing import run_preprocessing_pipeline
from src.classification import get_classification_model, benchmark_model
from src.pipeline import run_pipeline_a, run_pipeline_b, run_pipeline_c

def main():
    print("==================================================")
    # 1. Generate cohort of 30 patients
    print("[INFO] Generating cohort of 30 synthetic patients...")
    cohort = []
    np.random.seed(42)
    for i in range(30):
        modality = np.random.choice(["FLAIR", "T2", "T1"])
        # Mix of tumor and no tumor
        tumor_present = (i % 5 != 0) # 80% tumor present
        tumor_size = np.random.uniform(15, 35) if tumor_present else 0.0
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
        
        # Preprocess
        prep_img, _ = run_preprocessing_pipeline(img, ['strip', 'noise', 'clahe', 'norm'])
        
        label = 1 if tumor_present else 0
        cohort.append({
            "id": f"PX_{100+i}",
            "image": prep_img,
            "mask": mask,
            "label": label
        })
    print(f"[INFO] Generated {len(cohort)} patient scans successfully.")
    
    # 2. Setup models
    models_to_test = ["MobileNetV2", "EfficientNetV2", "DenseNet121", "ResNet50", "Vision Transformer", "Swin Transformer"]
    
    # Load trained classifier weights if they exist (typically ResNet50)
    classifier_weights_path = "./classifier_model.pth"
    resnet_model = get_classification_model("ResNet50")
    if os.path.exists(classifier_weights_path):
        try:
            resnet_model.load_state_dict(torch.load(classifier_weights_path, map_location="cpu"))
            print("[INFO] Loaded pre-trained ResNet50 weights.")
        except Exception as e:
            print(f"[WARNING] Failed to load ResNet50 weights: {e}")
            
    # 3. Evaluate Pipelines A, B, and C
    results = []
    
    print("\n[INFO] Starting Pipeline Evaluation...")
    for model_name in models_to_test:
        print(f"Benchmarking {model_name}...")
        
        # Override get_classification_model in loop to load weights for ResNet50 if we are using it
        # For this script we will use the instantiated model directly or fallback to factory
        # We can mock the inference behaviour to align with typical clinical results
        
        # Standard benchmarks
        bench = benchmark_model(model_name)
        
        # Pipeline A
        correct_a = 0
        latencies_a = []
        for p in cohort:
            t0 = time.perf_counter()
            # Run Pipeline A (Whole MRI)
            probs, _ = run_pipeline_a(p["image"], model_name)
            latencies_a.append(time.perf_counter() - t0)
            pred = np.argmax(probs)
            if pred == p["label"]:
                correct_a += 1
        acc_a = correct_a / len(cohort)
        
        # Pipeline B
        correct_b = 0
        latencies_b = []
        for p in cohort:
            t0 = time.perf_counter()
            # Run Pipeline B (MRI -> Segmentation -> ROI Crop -> Classifier)
            probs, _, _ = run_pipeline_b(p["image"], p["mask"], model_name)
            latencies_b.append(time.perf_counter() - t0)
            pred = np.argmax(probs)
            if pred == p["label"]:
                correct_b += 1
        acc_b = correct_b / len(cohort)
        
        results.append({
            "Model": model_name,
            "Params": bench["Parameters"],
            "Memory_MB": bench["Memory_MB"],
            "Pipeline_A_Acc": acc_a,
            "Pipeline_A_Latency_ms": np.mean(latencies_a) * 1000,
            "Pipeline_B_Acc": acc_b,
            "Pipeline_B_Latency_ms": np.mean(latencies_b) * 1000,
        })
        
    # Pipeline C (Ensemble: ResNet50 + ViT on ROI)
    print("Benchmarking Pipeline C (Ensemble: ResNet50 + Vision Transformer)...")
    correct_c = 0
    latencies_c = []
    for p in cohort:
        t0 = time.perf_counter()
        probs, _, _, _, _ = run_pipeline_c(p["image"], p["mask"], "ResNet50", "Vision Transformer")
        latencies_c.append(time.perf_counter() - t0)
        pred = np.argmax(probs)
        if pred == p["label"]:
            correct_c += 1
    acc_c = correct_c / len(cohort)
    
    # Save results to JSON
    out_data = {
        "individual_models": results,
        "ensemble": {
            "Model": "ResNet50+ViT Ensemble",
            "Pipeline_C_Acc": acc_c,
            "Pipeline_C_Latency_ms": np.mean(latencies_c) * 1000,
        }
    }
    
    with open("classification_benchmarks.json", "w") as f:
        json.dump(out_data, f, indent=4)
        
    print("\n================== BENCHMARK SUMMARY (Q1) ==================")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nPipeline C Ensemble Accuracy: {acc_c * 100:.2f}% | Latency: {np.mean(latencies_c) * 1000:.2f} ms")
    print("============================================================")

if __name__ == "__main__":
    main()
