# PrognosAI-X: Explainable Brain Tumor Decision Support Platform

PrognosAI-X is an explainable multi-stage deep learning platform designed for brain tumor detection, voxel-level segmentation, classification, clinical risk assessment, and automated radiologist report generation. 

It is engineered as a research prototype suitable for IEEE conference submission, biomedical engineering senior design, and clinical AI demonstrations.

---

## 🚀 Key Framework Pillars

### 1. Advanced Signal & Image Preprocessing
Implements a standardized, reproducible preprocessing pipeline to normalize and clean raw MRI scans:
- **Active Contour Skull Stripping:** Eliminates non-brain structures (scalp, skull bone) using threshold-guided morphology.
- **Bilateral Noise Filtering:** Edge-preserving smoothing that attenuates thermal scanner noise.
- **CLAHE Contrast Modification:** Locally equalizes intensity histograms to highlight soft-tissue tumor margins.
- **Z-score Intensity Normalization:** Standardizes intensity distributions across scans:
  $$I_{\text{norm}}(x, y) = \frac{I(x, y) - \mu_{\text{active}}}{\sigma_{\text{active}}}$$
- **Image Registration:** Co-registers misaligned slices using keypoint feature matches (ORB) and RANSAC homography.

### 2. Multi-Stage Segmentation & ROI Crop
Delineates precise tumor margins and performs region-of-interest extraction:
- **Architectures:** Implements U-Net, Attention U-Net (with additive soft attention gates), U-Net++, and Mask R-CNN.
- **Metrics:** Computes Dice Similarity Coefficient, Intersection over Union (IoU), Boundary F1, Precision, Recall, and Hausdorff Distance.
- **ROI Extraction:** Dynamically crops the segmented tumor area with safety padding to preserve edge margins for local pathology classification.

### 3. ROI-Guided Ensemble Classification
Evaluates pathological malignancy profiles using deep architectures:
- **Classification Backbones:** Compares MobileNetV2, EfficientNetV2, DenseNet121, ResNet50, Vision Transformer (ViT), and Swin Transformer.
- **Pipeline Comparison:**
  - *Pipeline A:* Whole MRI -> Classifier
  - *Pipeline B:* MRI -> Segmentation -> ROI Crop -> Classifier
  - *Pipeline C (Ours):* MRI -> Segmentation -> ROI Crop -> CNN + ViT Ensemble (soft-voting average)
- **Scientific recommendation:** Outperforms whole-brain classifiers by isolating parameters onto lesion textures, validated via ANOVA.

### 4. Dual-Gradient and Perturbation Explainability (XAI)
Establishes clinician trust by visualizing model parameters side-by-side:
- **Methods:** Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, and Kernel SHAP.
- **Focus Quality Classifier:** Measures IoU overlap between XAI heatmaps and the segmentation mask to automatically classify focus as *Correct*, *Partially Correct*, or *Incorrect*.
- **Explainability Confidence Score:**
  $$\text{Score} = 0.4 \times \text{Overlap IoU} + 0.6 \times \text{Prediction Probability}$$

### 5. Quantitative Biomarkers & Risk Profiling
Extracts physical and texture tumor parameters:
- **Biomarkers:** Tumor Area ($mm^2$), Perimeter ($mm$), Circularity (Irregularity), Relative Tissue Density, and GLCM (Gray-Level Co-occurrence Matrix) texture features (Contrast, Correlation, Energy, Homogeneity).
- **Risk Assessment:** Automates patient risk categorization (Low, Moderate, High Risk) with detailed radiologist recommendations.

### 6. Automated LLM Radiology Reporting
Integrates the Gemini API to synthesize quantitative biomarkers, classification predictions, and segmentation metrics into a standardized radiological clinical report with a mandatory medical disclaimer. Includes a local high-fidelity markdown template generator fallback for offline use.

---

## 📁 Repository Directory Structure

```
d:\BME\CEP_PBL_OEL\
├── .venv/                         # Local Python Virtual Environment
├── requirements.txt               # Frozen dependencies list
├── README.md                      # Platform documentation (This file)
├── run.ps1                        # PowerShell launch script
├── mlflow_local_runs.json         # Local MLOps backup runs database
├── src/
│   ├── generator.py               # Synthetic T1/T2/FLAIR MRI and tumor simulator
│   ├── preprocessing.py           # Skull stripping, registration, and normalizations
│   ├── segmentation.py            # U-Net, Attention U-Net, U-Net++, and metrics
│   ├── classification.py          # MobileNetV2, ResNet50, Swin, ViT backbones & profiling
│   ├── pipeline.py                # Pipelines A, B, and C workflow logic
│   ├── explainability.py          # Grad-CAM, Score-CAM, IG, SHAP, and Focus Quality
│   ├── statistics.py              # ANOVA, t-tests, McNemar, Cohen's d, Bootstrap
│   ├── risk_assessment.py         # Biomarker extraction and risk scoring
│   ├── llm_report.py              # Gemini API report generator
│   └── mlflow_tracker.py          # Experiment run logger (MLflow / JSON)
├── app/
│   └── dashboard.py               # 10-tab Streamlit medical workstation UI
└── tests/
    ├── test_preprocessing.py      # Preprocessing unit tests
    ├── test_segmentation.py       # Segmentation architectures & metrics unit tests
    └── test_statistics.py         # Hypothesis testing unit tests
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the repository directory:**
   ```bash
   cd d:\BME\CEP_PBL_OEL
   ```

2. **Initialize and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install frozen dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Launching the Workstation

To start the Streamlit server and load the workstation dashboard in your browser:

```powershell
.\run.ps1
```

If you do not have PowerShell execution permissions, activate your environment and execute Streamlit directly:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/dashboard.py
```

---

## 🧪 Running Unit Tests

Verify the computational blocks and models using pytest:

```powershell
pytest tests/
```
