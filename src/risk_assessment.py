import numpy as np
import cv2

# =====================================================================
# 1. Custom NumPy-based GLCM Texture Engine (Safe Offline Fallback)
# =====================================================================

def compute_custom_glcm(image, mask, num_levels=8, distance=1, angle=0):
    """
    Computes a Gray-Level Co-occurrence Matrix (GLCM) inside the tumor mask.
    Quantizes intensities to 'num_levels' grey levels.
    
    Returns:
        contrast (float)
        correlation (float)
        energy (float)
        homogeneity (float)
    """
    # 1. Crop bounding box of tumor to save computation
    pts = np.argwhere(mask > 0.5)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    
    crop_img = image[y_min:y_max+1, x_min:x_max+1]
    crop_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    # 2. Quantize crop image into [0, num_levels-1]
    # Background pixels inside mask only
    active_img = crop_img.copy()
    active_img[crop_mask <= 0.5] = 0.0
    
    v_max = active_img.max()
    v_min = active_img[crop_mask > 0.5].min() if np.any(crop_mask > 0.5) else 0.0
    
    if v_max == v_min:
        return 0.0, 0.0, 1.0, 1.0
        
    quantized = np.zeros_like(active_img, dtype=np.int32)
    mask_indices = crop_mask > 0.5
    quantized[mask_indices] = np.clip(
        ((active_img[mask_indices] - v_min) / (v_max - v_min) * (num_levels - 1)).astype(np.int32),
        0, num_levels - 1
    )
    
    # 3. Fill GLCM Matrix
    glcm = np.zeros((num_levels, num_levels), dtype=np.float32)
    h, w = quantized.shape
    
    # Offset based on angle
    # angle=0 (horizontal right): dy=0, dx=1
    # angle=90 (vertical down): dy=1, dx=0
    # angle=45 (diagonal up-right): dy=-1, dx=1
    # angle=135 (diagonal up-left): dy=-1, dx=-1
    if angle == 0:
        dy, dx = 0, distance
    elif angle == 90:
        dy, dx = distance, 0
    elif angle == 45:
        dy, dx = -distance, distance
    else:  # 135
        dy, dx = -distance, -distance
        
    count = 0
    for r in range(h):
        for c in range(w):
            if crop_mask[r, c] <= 0.5:
                continue
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                if crop_mask[nr, nc] > 0.5:
                    i = quantized[r, c]
                    j = quantized[nr, nc]
                    glcm[i, j] += 1
                    count += 1
                    
    if count == 0:
        return 0.0, 0.0, 1.0, 1.0
        
    # Normalize GLCM to probabilities
    glcm = glcm / count
    
    # Symmetric GLCM
    glcm = (glcm + glcm.T) / 2.0
    
    # 4. Compute Texture Properties
    contrast = 0.0
    energy = 0.0
    homogeneity = 0.0
    
    # Means and standard deviations for correlation
    i_indices = np.arange(num_levels)
    j_indices = np.arange(num_levels)
    
    marginal_i = np.sum(glcm, axis=1)
    marginal_j = np.sum(glcm, axis=0)
    
    mean_i = np.sum(i_indices * marginal_i)
    mean_j = np.sum(j_indices * marginal_j)
    
    var_i = np.sum((i_indices - mean_i) ** 2 * marginal_i)
    var_j = np.sum((j_indices - mean_j) ** 2 * marginal_j)
    
    std_i = np.sqrt(var_i)
    std_j = np.sqrt(var_j)
    
    correlation = 0.0
    
    for i in range(num_levels):
        for j in range(num_levels):
            val = glcm[i, j]
            if val == 0:
                continue
            contrast += val * ((i - j) ** 2)
            energy += val ** 2
            homogeneity += val / (1.0 + abs(i - j))
            
            if std_i > 0 and std_j > 0:
                correlation += val * (i - mean_i) * (j - mean_j) / (std_i * std_j)
                
    return float(contrast), float(correlation), float(energy), float(homogeneity)

# =====================================================================
# 2. Shape Biomarker Extraction & Risk Assessor
# =====================================================================

def extract_biomarkers(image, mask, pixel_spacing_mm=0.5):
    """
    Extracts clinical shape and texture biomarkers from an MRI slice and segmentation mask.
    pixel_spacing_mm: size of one pixel side (default 0.5mm, typical MRI slice resolution)
    """
    m_u8 = (mask > 0.5).astype(np.uint8)
    
    # 1. Shape Features
    # Area (pixel count scaled to mm2)
    pixel_area_mm2 = pixel_spacing_mm ** 2
    tumor_pixels = np.sum(m_u8)
    area = float(tumor_pixels * pixel_area_mm2)
    
    # Find contours for perimeter
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        perimeter = 0.0
        circularity = 1.0
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        # Perimeter in mm
        perimeter = float(cv2.arcLength(largest_contour, closed=True) * pixel_spacing_mm)
        
        # Circularity (Shape Irregularity)
        if perimeter == 0:
            circularity = 1.0
        else:
            circularity = float((4 * np.pi * area) / (perimeter ** 2))
            
    # Clip circularity to [0, 1]
    circularity = min(1.0, max(0.0, circularity))
    
    # 2. Intensity/Density Features
    non_zero_brain = image[image > 0]
    healthy_mean = np.mean(non_zero_brain) if len(non_zero_brain) > 0 else 0.5
    
    tumor_pixels_val = image[mask > 0.5]
    if len(tumor_pixels_val) == 0:
        tumor_mean = 0.0
        tumor_density = 0.0
    else:
        tumor_mean = np.mean(tumor_pixels_val)
        tumor_density = float(tumor_mean / healthy_mean) # Ratio to normal brain tissue
        
    # 3. Texture Features
    # Compute custom GLCM averages over two directions (0 and 90 degrees)
    cont0, corr0, ene0, hom0 = compute_custom_glcm(image, mask, angle=0)
    cont90, corr90, ene90, hom90 = compute_custom_glcm(image, mask, angle=90)
    
    contrast = (cont0 + cont90) / 2.0
    correlation = (corr0 + corr90) / 2.0
    energy = (ene0 + ene90) / 2.0
    homogeneity = (hom0 + hom90) / 2.0
    
    return {
        "Tumor Area (mm2)": area,
        "Tumor Perimeter (mm)": perimeter,
        "Circularity (Shape Regularity)": circularity,
        "Relative Tumor Density": tumor_density,
        "GLCM Contrast (Heterogeneity)": contrast,
        "GLCM Correlation": correlation,
        "GLCM Energy (Uniformity)": energy,
        "GLCM Homogeneity": homogeneity
    }

def assess_clinical_risk(biomarkers, prediction_prob):
    """
    Estimates Patient Clinical Risk based on biomarker thresholds and prediction scores.
    """
    area = biomarkers["Tumor Area (mm2)"]
    circularity = biomarkers["Circularity (Shape Regularity)"]
    contrast = biomarkers["GLCM Contrast (Heterogeneity)"]
    density = biomarkers["Relative Tumor Density"]
    
    # Risk scoring algorithm
    risk_score = 0.0
    
    # Area contribution
    if area > 1200:
        risk_score += 3.0
    elif area > 400:
        risk_score += 1.5
    else:
        risk_score += 0.5
        
    # Circularity (lower is more irregular/infiltrating)
    if circularity < 0.45:
        risk_score += 3.0
    elif circularity < 0.65:
        risk_score += 1.5
    else:
        risk_score += 0.5
        
    # Contrast / Heterogeneity (necrotic cores have higher contrast)
    if contrast > 1.2:
        risk_score += 2.0
    elif contrast > 0.5:
        risk_score += 1.0
    else:
        risk_score += 0.2
        
    # Relative density (high-grade tumors are hyperdense/enhancing)
    if density > 1.4 or density < 0.6:
        risk_score += 2.0
    else:
        risk_score += 1.0
        
    # Map to Risk Categories
    if risk_score >= 7.5:
        category = "Grade 4 Glioma"
        description = (
            "Large, highly irregular tumor boundary with micro-texture heterogeneity. "
            "Suggestive of a high-grade aggressive neoplasm (e.g., Glioblastoma Multiforme, WHO Grade 4) "
            "with surrounding mass effect and active infiltration edema."
        )
    elif risk_score >= 4.0:
        category = "Grade 2-3 Glioma"
        description = (
            "Medium-sized tumor with moderate boundary irregularity. "
            "Suggestive of a mid-grade glioma (WHO Grade 2-3) or active pituitary macro-adenoma. "
            "Requires immediate radiological follow-up and contrast-enhanced study."
        )
    else:
        category = "Grade 1 Glioma / Benign"
        description = (
            "Small, encapsulated lesion with high circularity. "
            "Consistent with a benign slow-growing neoplasm (e.g., Meningioma, WHO Grade 1, or vestibular schwannoma). "
            "Observation with periodic MRI surveillance or stereotactic radiosurgery might be indicated."
        )
        
    # Risk Confidence calculation
    risk_confidence = float(0.7 * prediction_prob + 0.3 * circularity)
    
    return {
        "Risk Category": category,
        "Risk Score": float(risk_score),
        "Clinical Description": description,
        "Confidence": risk_confidence
    }

