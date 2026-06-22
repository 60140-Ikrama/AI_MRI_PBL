import numpy as np
import cv2

def generate_synthetic_slice(
    modality="FLAIR",
    tumor_present=True,
    tumor_size=30,
    tumor_loc=(0.1, -0.15),  # relative coordinates from center (-0.5 to 0.5)
    noise_level=0.03,
    bias_field=True,
    misalign_x=0.0,
    misalign_y=0.0,
    misalign_rot=0.0,
    seed=None
):
    """
    Generates a publication-quality synthetic brain MRI slice with customizable tumor,
    noise, bias field, and spatial misalignment for testing registration and preprocessing.
    
    Returns:
        image (np.ndarray): 256x256 image normalized to [0, 1]
        mask (np.ndarray): 256x256 binary mask of the tumor
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = 256, 256
    img = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    
    center_y, center_x = h // 2, w // 2
    
    # 1. Base Tissue Intensities (T1, T2, FLAIR)
    if modality == "T1":
        bg_val = 0.0
        skull_val = 0.8
        csf_val = 0.15
        grey_val = 0.45
        white_val = 0.70
        tumor_val = 0.35  # T1 tumor is typically hypointense
    elif modality == "T2":
        bg_val = 0.0
        skull_val = 0.4
        csf_val = 0.95  # T2 CSF is hyperintense
        grey_val = 0.60
        white_val = 0.40
        tumor_val = 0.85  # T2 tumor is hyperintense
    else:  # FLAIR
        bg_val = 0.0
        skull_val = 0.5
        csf_val = 0.05  # FLAIR CSF is suppressed
        grey_val = 0.55
        white_val = 0.40
        tumor_val = 0.95  # FLAIR tumor is hyperintense
        
    # 2. Draw Skull (outer bone ellipse)
    cv2.ellipse(img, (center_x, center_y), (100, 120), 0, 0, 360, skull_val, -1)
    # Inner skull (bone marrow / details)
    cv2.ellipse(img, (center_x, center_y), (97, 117), 0, 0, 360, skull_val * 0.6, -1)
    
    # 3. Draw Brain Parenchyma (dura matter boundary)
    brain_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(brain_mask, (center_x, center_y), (90, 110), 0, 0, 360, 1.0, -1)
    
    # Create grey matter / white matter structure inside brain mask
    # White matter core
    wm_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(wm_mask, (center_x, center_y), (60, 80), 0, 0, 360, 1.0, -1)
    # Give it some lobulated structure using small overlapping ellipses
    for angle in range(0, 360, 45):
        rad = np.deg2rad(angle)
        offset_x = int(50 * np.cos(rad))
        offset_y = int(70 * np.sin(rad))
        cv2.ellipse(wm_mask, (center_x + offset_x, center_y + offset_y), (25, 25), 0, 0, 360, 1.0, -1)
        
    # Intersect with brain mask to keep it inside the skull
    wm_mask = cv2.bitwise_and(wm_mask, brain_mask)
    
    # Assign intensities to parenchyma
    # Grey matter is brain minus white matter
    gm_mask = cv2.subtract(brain_mask, wm_mask)
    
    img[gm_mask > 0] = grey_val
    img[wm_mask > 0] = white_val
    
    # Add ventricles (bilateral dark structures)
    ventricle_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(ventricle_mask, (center_x - 20, center_y - 15), (12, 35), -15, 0, 360, 1.0, -1)
    cv2.ellipse(ventricle_mask, (center_x + 20, center_y - 15), (12, 35), 15, 0, 360, 1.0, -1)
    
    img[ventricle_mask > 0] = csf_val
    
    # Background outside skull is 0
    img[img == bg_val] = 0.0
    
    # 4. Generate Tumor (if present)
    if tumor_present and tumor_size > 0:
        # Calculate pixel position of tumor center
        t_cy = int(center_y + tumor_loc[0] * h)
        t_cx = int(center_x + tumor_loc[1] * w)
        
        # Create an irregular polygon for the tumor
        num_vertices = 12
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        # Randomize radii to create irregular shape
        radii = tumor_size * (0.85 + 0.3 * np.random.randn(num_vertices))
        # Keep radii bounded
        radii = np.clip(radii, tumor_size * 0.4, tumor_size * 1.6)
        
        points = []
        for a, r in zip(angles, radii):
            px = int(t_cx + r * np.cos(a))
            py = int(t_cy + r * np.sin(a))
            points.append([px, py])
            
        pts = np.array(points, dtype=np.int32)
        
        # Draw tumor on separate mask
        t_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(t_mask, [pts], 1.0)
        
        # Keep tumor inside the brain parenchyma boundary
        t_mask = cv2.bitwise_and(t_mask, brain_mask)
        
        # Save mask
        mask[t_mask > 0] = 1.0
        
        # Apply tumor intensity inside brain
        # Blend tumor tissue realistically
        img[mask > 0] = tumor_val
        
        # Give tumor a necrosis core (darker inside)
        if tumor_size > 15:
            core_mask = np.zeros((h, w), dtype=np.float32)
            pts_float = np.array(points, dtype=np.float32)
            cv2.fillPoly(core_mask, [np.array(pts_float * 0.45 + np.array([t_cx, t_cy]) * 0.55, dtype=np.int32)], 1.0)
            core_mask = cv2.bitwise_and(core_mask, mask)
            img[core_mask > 0] = tumor_val * 0.4
            
        # Give tumor an active edema border (hyperintense ring in T2/FLAIR)
        if modality in ["T2", "FLAIR"]:
            edema_mask = cv2.dilate(t_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            edema_mask = cv2.subtract(edema_mask, t_mask)
            edema_mask = cv2.bitwise_and(edema_mask, brain_mask)
            # Blend edema into image
            img[edema_mask > 0] = img[edema_mask > 0] * 0.4 + 0.75 * 0.6
            
    # Suppress anything outside the outer head boundary just in case
    head_boundary = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(head_boundary, (center_x, center_y), (105, 125), 0, 0, 360, 1.0, -1)
    img[head_boundary == 0] = 0.0
    mask[head_boundary == 0] = 0.0
    
    # 5. Apply Bias Field Inhomogeneity
    if bias_field:
        y_coords, x_coords = np.mgrid[-1:1:256j, -1:1:256j]
        # Quadratic bias field: lower signal in bottom-right, higher in top-left
        bias = 1.0 + 0.15 * x_coords + 0.1 * y_coords - 0.1 * (x_coords**2 + y_coords**2)
        img = img * bias
        
    # Clip image values before noise
    img = np.clip(img, 0, 1)
    
    # 6. Apply Noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, (h, w)).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0, 1)
        
    # 7. Apply Spatial Misalignment (Translation + Rotation)
    if abs(misalign_x) > 0 or abs(misalign_y) > 0 or abs(misalign_rot) > 0:
        M = cv2.getRotationMatrix2D((center_x, center_y), misalign_rot, 1.0)
        M[0, 2] += misalign_x
        M[1, 2] += misalign_y
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
    return img, mask
