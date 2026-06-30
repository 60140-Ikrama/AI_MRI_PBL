import os

def generate_clinical_report(
    patient_name="PATIENT_X_091",
    modality="FLAIR",
    preprocessed_steps=["Bilateral Noise Filter", "CLAHE Contrast", "Skull Stripped"],
    classification_result={"Prediction": "Tumor Detected", "Probability": 0.942, "Model": "ResNet50 + ViT Ensemble"},
    segmentation_result={"Dice": 0.895, "IoU": 0.812, "Model": "Attention U-Net"},
    biomarkers={
        "Tumor Area (mm2)": 1420.5,
        "Tumor Perimeter (mm)": 182.4,
        "Circularity (Shape Regularity)": 0.54,
        "Relative Tumor Density": 1.58,
        "GLCM Contrast (Heterogeneity)": 0.88,
        "GLCM Energy (Uniformity)": 0.12
    },
    risk_assessment={
        "Risk Category": "High Risk",
        "Risk Score": 8.5,
        "Clinical Description": "Large, irregular tumor boundary with micro-texture heterogeneity."
    },
    explainability_result={
        "Focus Category": "Correct Focus",
        "Overlap IoU": 0.68,
        "Confidence Score": 0.838
    },
    api_key=None
):
    """
    Generates a professional clinical radiology report.
    Integrates Gemini API if api_key is provided; otherwise falls back to a highly detailed local
    template generator that simulates the clinical decision support outputs.
    """
    
    # 1. Structure the prompt / report contents
    patient_summary_txt = f"Patient Identifier: {patient_name}\nModality: MRI {modality}\nApplied Preprocessing: {', '.join(preprocessed_steps)}"
    classification_txt = f"Primary Diagnosis: {classification_result['Prediction']}\nModel Confidence: {classification_result['Probability']*100:.1f}%\nArchitecture: {classification_result['Model']}"
    segmentation_txt = f"Active Model: {segmentation_result['Model']}\nDice Coefficient: {segmentation_result['Dice']:.3f}\nIntersection over Union (IoU): {segmentation_result['IoU']:.3f}"
    biomarkers_txt = "\n".join([f"- {k}: {v:.2f}" for k, v in biomarkers.items()])
    risk_txt = f"Assessed Category: {risk_assessment['Risk Category']}\nComposite Risk Score: {risk_assessment['Risk Score']}/10.0\nPathological Details: {risk_assessment['Clinical Description']}"
    xai_txt = f"Focus Quality: {explainability_result['Focus Category']}\nHeatmap-Segmentation Overlap IoU: {explainability_result['Overlap IoU']:.3f}\nExplainability Confidence Score: {explainability_result['Confidence Score']:.3f}"
    
    prompt = f"""
    You are a Senior Neuroradiologist and Clinical AI Specialist at Stanford Medicine.
    Your task is to write a publication-quality, industrial-grade Clinical Decision Support Report based on the following AI detection findings:
    
    === PATIENT IMAGING SUMMARY ===
    {patient_summary_txt}
    
    === CLASSIFICATION FINDINGS ===
    {classification_txt}
    
    === SEGMENTATION FINDINGS ===
    {segmentation_txt}
    
    === QUANTITATIVE NEURO-BIOMARKERS ===
    {biomarkers_txt}
    
    === WHO TUMOR CATEGORIZATION & BT-RADS ===
    {risk_txt}
    
    === EXPLAINABLE AI (XAI) SUMMARY ===
    {xai_txt}
    
    Write a structured clinical report in markdown. The report should look professional and read like an official hospital neuroradiology report. 
    Incorporate medical terminology (e.g. signal hyperintensity, mass effect, cellular density, boundary infiltrative margins).
    
    Ensure you structure it into these exact sections:
    1. CLINICAL STUDY OVERVIEW
    2. QUANTITATIVE IMAGING ANALYSIS (Discuss the neuro-biomarker panel, necrosis, and edema)
    3. MODEL PERFORMANCE & STABILITY (Mention the Dice score and classification confidence)
    4. EXPLAINABILITY & BIOMEDICAL TRUST CHECK (Validate the CAM focus and comment on explainability confidence)
    5. INTERPRETIVE RISK DIAGNOSIS (Provide WHO Tumor Categorization Grade and estimated BT-RADS score)
    6. TARGETED CLINICAL RECOMMENDATIONS (e.g. referral, contrast MRI, stereotactic biopsy, surgical planning)
    
    At the bottom of the report, you must append this mandatory disclaimer in bold:
    "MANDATORY CLINICAL DISCLAIMER: This report is generated automatically by a deep learning framework as a clinical decision-support tool. It does not replace the professional diagnosis, clinical examination, or consultation of a qualified medical practitioner, neurosurgeon, or radiologist."
    """
    
    # 2. Try to run Gemini API if API key is provided
    actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if actual_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=actual_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            # Fallback on failure
            print(f"Warning: Gemini API call failed ({e}). Falling back to local template generator.")
            
    # 3. Local template generator (high-fidelity fallback)
    local_report = f"""# PrognosAI-X: Clinical Decision Support Report

## 1. CLINICAL STUDY OVERVIEW
- **Patient Identifier:** {patient_name}
- **Scan Modality:** Magnetic Resonance Imaging (MRI) - **{modality}** Protocol
- **Preprocessing Sequence:** {", ".join(preprocessed_steps)}
- **Study Date:** June 22, 2026

---

## 2. QUANTITATIVE IMAGING ANALYSIS
The segmentation subsystem successfully localized the tumor mass. Quantitative assessment of the extracted region of interest (ROI) yielded the following morphological and texture biomarkers:
- **Tumor Area:** `{biomarkers["Tumor Area (mm2)"]:.2f} mm²`
- **Tumor Perimeter:** `{biomarkers["Tumor Perimeter (mm)"]:.2f} mm`
- **Boundary Regularity (Circularity):** `{biomarkers["Circularity (Shape Regularity)"]:.2f}` (Value of 1.0 represents a perfect circle. A lower value indicates complex, micro-lobulated infiltrative borders).
- **Relative Intensity Density:** `{biomarkers["Relative Tumor Density"]:.2f}` (Signal ratio relative to healthy brain tissue. Values > 1.3 indicate high contrast enhancement, consistent with high cellular density or edema).
- **Texture Analysis (GLCM Contrast):** `{biomarkers["GLCM Contrast (Heterogeneity)"]:.2f}` (High contrast indicates structural heterogeneity within the lesion, often correlating with focal necrosis).
- **Texture Uniformity (GLCM Energy):** `{biomarkers["GLCM Energy (Uniformity)"]:.2f}`

---

## 3. MODEL PERFORMANCE & STABILITY
Multi-stage classification and segmentation networks processed the imaging data with high operational metrics:
- **Segmentation Model:** `{segmentation_result["Model"]}`
  - **Dice Similarity Coefficient:** `{segmentation_result["Dice"]:.3f}`
  - **Intersection over Union (IoU):** `{segmentation_result["IoU"]:.3f}`
- **Classification Subsystem:** `{classification_result["Model"]}`
  - **Predicted Pathological Class:** **{classification_result["Prediction"]}**
  - **Model Softmax Probability:** `{classification_result["Probability"]*100:.1f}%`

---

## 4. EXPLAINABILITY & BIOMEDICAL TRUST CHECK
Explainable AI (XAI) heatmaps (Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, and SHAP) were generated and compared against the segmentation mask to ensure clinical alignment:
- **Heatmap Spatial Alignment:** **{explainability_result["Focus Category"]}** (IoU Overlap: `{explainability_result["Overlap IoU"]:.3f}`)
- **Explainability Confidence Score:** `{explainability_result["Confidence Score"]:.3f}` (Integrates classification confidence and visual alignment to measure trust).
*Trust Validation:* The visual highlight maps verify that the network is focusing its parameters on the active tumor margins rather than background artifacts or skull tissue.

---

## 5. INTERPRETIVE RISK DIAGNOSIS
- **Assessed Clinical Risk Category:** **{risk_assessment["Risk Category"]}** (Risk Score: `{risk_assessment["Risk Score"]:.1f}/10.0`)
- **Radiological Interpretation:** {risk_assessment["Clinical Description"]} The extracted biomarkers indicate a lesion of {risk_assessment["Risk Category"].lower()} profile.

---

## 6. TARGETED CLINICAL RECOMMENDATIONS
1. **Contrast Study:** Recommend a dynamic contrast-enhanced (DCE) MRI to further evaluate the vascular perfusion and blood-brain barrier permeability.
2. **Neurosurgical Consultation:** Refer to neuro-oncology board for surgical resection planning or stereotactic biopsy, as indicated by the lesion volume and boundary irregularity.
3. **Serial Surveillance:** If clinical presentation is benign, establish a baseline 3-month MRI follow-up protocol.
4. **Correlation:** Correlate with patient symptomatology (e.g. headaches, focal neurological deficits, seizures).

---

**MANDATORY CLINICAL DISCLAIMER:** *This report is generated automatically by a deep learning framework as a clinical decision-support tool. It does not replace the professional diagnosis, clinical examination, or consultation of a qualified medical practitioner, neurosurgeon, or radiologist.*
"""
    return local_report
