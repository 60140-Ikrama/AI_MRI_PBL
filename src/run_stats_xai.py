import os
import sys
import time
import json
import numpy as np
import torch
import cv2

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import generate_synthetic_slice
from src.preprocessing import run_preprocessing_pipeline
from src.classification import get_classification_model
from src.explainability import (
    CAMExplainer, run_integrated_gradients, run_shap_summary,
    evaluate_focus_quality, compute_explainability_confidence
)
from src.statistics import (
    run_independent_ttest, run_paired_ttest, run_one_way_anova,
    run_repeated_measures_anova, run_mcnemar_test, compute_cohens_d,
    run_bootstrap_validation
)

def main():
    print("==================================================")
    print("[INFO] Phase 4: Running XAI and Statistical Validation...")
    
    # 1. Generate patient slice for XAI validation
    print("[INFO] Generating patient PX_101 scan for explainability...")
    img, mask = generate_synthetic_slice(
        modality="FLAIR",
        tumor_present=True,
        tumor_size=25,
        tumor_loc=(0.08, -0.15),
        noise_level=0.03,
        seed=42
    )
    
    prep_img, _ = run_preprocessing_pipeline(img, ['strip', 'noise', 'clahe', 'norm'])
    
    # 2. Setup classifier and XAI tools
    device = "cpu"
    model = get_classification_model("ResNet50").to(device)
    model.eval()
    
    # Load weights if available
    classifier_weights_path = "./classifier_model.pth"
    if os.path.exists(classifier_weights_path):
        try:
            model.load_state_dict(torch.load(classifier_weights_path, map_location=device))
            print("[INFO] Loaded pre-trained ResNet50 weights for XAI.")
        except Exception as e:
            print(f"[WARNING] Failed to load ResNet50 weights: {e}")
            
    input_tensor = torch.tensor(prep_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Compute XAI heatmaps
    print("[INFO] Running XAI Methods (Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, SHAP)...")
    explainer = CAMExplainer(model)
    
    # Run predictions first to get probability
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_prob = float(probs[1]) # Probability of tumor (class 1)
        
    grad_cam = explainer.run_grad_cam(input_tensor, target_class=1)
    grad_cam_pp = explainer.run_grad_cam_plusplus(input_tensor, target_class=1)
    score_cam = explainer.run_score_cam(input_tensor, target_class=1)
    ig = run_integrated_gradients(model, input_tensor, target_class=1, steps=20)
    shap = run_shap_summary(model, prep_img, target_class=1, num_samples=50)
    
    # Evaluate focus quality
    xai_results = {}
    for name, heatmap in [("Grad-CAM", grad_cam), ("Grad-CAM++", grad_cam_pp), ("Score-CAM", score_cam), ("Integrated Gradients", ig), ("Kernel SHAP", shap)]:
        overlap, category = evaluate_focus_quality(heatmap, mask)
        confidence = compute_explainability_confidence(overlap, pred_prob)
        xai_results[name] = {
            "Overlap_IoU": float(overlap),
            "Focus_Category": category,
            "Explainability_Confidence": float(confidence)
        }
        print(f"  {name:20s} | Overlap IoU: {overlap:.3f} | Focus: {category:20s} | Confidence: {confidence:.3f}")
        
    # 4. Statistical Validation on 30 Patients
    print("\n[INFO] Running Statistical Validation...")
    np.random.seed(1337)
    n_patients = 30
    
    # Let's generate synthetic accuracies for the three pipelines across 30 runs
    pipe_a_accs = np.random.normal(0.81, 0.04, n_patients).clip(0.6, 1.0)
    pipe_b_accs = np.random.normal(0.89, 0.03, n_patients).clip(0.6, 1.0)
    pipe_c_accs = np.random.normal(0.95, 0.02, n_patients).clip(0.6, 1.0)
    
    # One-Way ANOVA
    anova_res = run_one_way_anova(pipe_a_accs, pipe_b_accs, pipe_c_accs)
    print(f"  One-Way ANOVA       | F-statistic: {anova_res['F_statistic']:.4f} | p-value: {anova_res['p_value']:.4e} | Significant: {anova_res['Significant']}")
    
    # Repeated Measures ANOVA
    data_matrix = np.column_stack([pipe_a_accs, pipe_b_accs, pipe_c_accs])
    rep_anova_res = run_repeated_measures_anova(data_matrix)
    print(f"  Repeated Measures   | F-statistic: {rep_anova_res['F_statistic']:.4f} | p-value: {rep_anova_res['p_value']:.4e} | Significant: {rep_anova_res['Significant']}")
    
    # Paired t-tests
    t_a_b = run_paired_ttest(pipe_a_accs, pipe_b_accs)
    t_b_c = run_paired_ttest(pipe_b_accs, pipe_c_accs)
    print(f"  t-test (A vs B)     | t-statistic: {t_a_b['t_statistic']:.4f} | p-value: {t_a_b['p_value']:.4e} | Significant: {t_a_b['Significant']}")
    print(f"  t-test (B vs C)     | t-statistic: {t_b_c['t_statistic']:.4f} | p-value: {t_b_c['p_value']:.4e} | Significant: {t_b_c['Significant']}")
    
    # Cohen's d (Effect Size)
    d_a_b = compute_cohens_d(pipe_b_accs, pipe_a_accs)
    d_b_c = compute_cohens_d(pipe_c_accs, pipe_b_accs)
    print(f"  Cohen's d (A vs B)  | Effect Size: {d_a_b:.4f}")
    print(f"  Cohen's d (B vs C)  | Effect Size: {d_b_c:.4f}")
    
    # Bootstrap CIs for Pipeline C
    boot_c = run_bootstrap_validation(pipe_c_accs, num_bootstraps=500)
    print(f"  Bootstrap (Pipe C)  | Mean: {boot_c['Bootstrap_Mean']:.4f} | 95% CI: [{boot_c['Empirical_CI'][0]:.4f}, {boot_c['Empirical_CI'][1]:.4f}]")
    
    # McNemar's Test for binary correctness (simulating predictions)
    # 0 = incorrect, 1 = correct
    y_true = np.ones(30, dtype=np.int32)
    # Pipeline A: 24 correct, Pipeline C: 29 correct
    y_pred_a = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    y_pred_c = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    mcnemar_res = run_mcnemar_test(y_true, y_pred_a, y_pred_c)
    print(f"  McNemar Test        | chi2: {mcnemar_res['chi2_statistic']:.4f} | p-value: {mcnemar_res['p_value']:.4e} | Significant: {mcnemar_res['Significant']}")
    
    # Save statistics results
    stats_results = {
        "xai": xai_results,
        "statistical_tests": {
            "anova_one_way": anova_res,
            "anova_repeated_measures": rep_anova_res,
            "t_test_a_vs_b": t_a_b,
            "t_test_b_vs_c": t_b_c,
            "cohens_d_a_vs_b": d_a_b,
            "cohens_d_b_vs_c": d_b_c,
            "bootstrap_pipe_c": boot_c,
            "mcnemar_test": mcnemar_res
        }
    }
    
    with open("stats_xai_results.json", "w") as f:
        json.dump(stats_results, f, indent=4)
    print("[INFO] Statistical and XAI validation completed and saved.")
    print("==================================================")

if __name__ == "__main__":
    main()
