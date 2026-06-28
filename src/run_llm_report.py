import os
import sys
import json

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_report import generate_clinical_report

def main():
    print("==================================================")
    print("[INFO] Phase 5: Generating AI Radiology Report...")
    
    # 1. Setup mock patient details based on high-risk profile
    patient_name = "PX_101"
    modality = "FLAIR"
    preprocessed_steps = ["Skull Stripping", "Bilateral Filtering", "CLAHE Enhancement", "Z-Score Normalization"]
    
    classification_result = {
        "Prediction": "Tumor Detected",
        "Probability": 0.946,
        "Model": "ResNet50 + Vision Transformer Ensemble (Pipeline C)"
    }
    
    segmentation_result = {
        "Dice": 0.895,
        "IoU": 0.812,
        "Model": "Attention U-Net"
    }
    
    biomarkers = {
        "Tumor Area (mm2)": 1420.50,
        "Tumor Perimeter (mm)": 182.40,
        "Circularity (Shape Regularity)": 0.54,
        "Relative Tumor Density": 1.58,
        "GLCM Contrast (Heterogeneity)": 0.88,
        "GLCM Energy (Uniformity)": 0.12
    }
    
    risk_assessment = {
        "Risk Category": "High Risk",
        "Risk Score": 8.50,
        "Clinical Description": "Large, highly irregular tumor boundary with micro-texture heterogeneity. Suggestive of high-grade aggressive neoplasm."
    }
    
    explainability_result = {
        "Focus Category": "Correct Focus",
        "Overlap IoU": 0.680,
        "Confidence Score": 0.838
    }
    
    # Generate report
    report_text = generate_clinical_report(
        patient_name=patient_name,
        modality=modality,
        preprocessed_steps=preprocessed_steps,
        classification_result=classification_result,
        segmentation_result=segmentation_result,
        biomarkers=biomarkers,
        risk_assessment=risk_assessment,
        explainability_result=explainability_result,
        api_key=None # Trigger local fallback template
    )
    
    # Save report
    with open("clinical_radiology_report_px101.md", "w") as f:
        f.write(report_text)
        
    print("[INFO] AI Radiology Report generated and saved to 'clinical_radiology_report_px101.md'.")
    print("==================================================")
    print(report_text[:400] + "...\n[TRUNCATED FOR DISPLAY]\n" + report_text[-300:])
    print("==================================================")

if __name__ == "__main__":
    main()
