import numpy as np
import cv2

def skull_strip(image, threshold_ratio=0.15):
    """
    Perform skull stripping on a brain MRI slice.
    Uses thresholding, morphological closing/opening, contour detection,
    and mask multiplication to isolate the brain parenchyma.
    """
    # 1. Convert to uint8 scale [0, 255] for OpenCV operations
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # 2. Thresholding: Otsu threshold to separate head from background
    _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological operations: close holes, open to detach skull
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 4. Find the largest contour (should be the outer head contour)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(opened)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
    # 5. Erode mask slightly to exclude skull bone details
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brain_mask = cv2.erode(mask, kernel_erode)
    
    # Convert mask to float [0.0, 1.0]
    brain_mask_f = brain_mask.astype(np.float32) / 255.0
    
    # Apply mask to strip skull
    stripped_image = image * brain_mask_f
    
    return stripped_image, brain_mask_f

def zscore_normalize(image):
    """
    Perform Z-score normalization based on non-zero pixels.
    Ensures background zeros do not skew the mean and standard deviation.
    """
    non_zero = image[image > 0]
    if len(non_zero) == 0:
        return image
    mean = np.mean(non_zero)
    std = np.std(non_zero)
    if std == 0:
        return image
    
    normalized = (image - mean) / std
    # Keep background as 0
    normalized[image == 0] = 0.0
    return normalized

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Increases local contrast while limiting noise amplification.
    """
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_u8)
    return enhanced.astype(np.float32) / 255.0

def reduce_noise_bilateral(image, d=5, sigma_color=0.08, sigma_space=8.0):
    """
    Reduces noise using a Bilateral Filter.
    Preserves edges while smoothing flat areas.
    """
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    # Convert parameters to uint8 scale
    sigma_color_u8 = int(sigma_color * 255)
    filtered = cv2.bilateralFilter(img_u8, d, sigma_color_u8, sigma_space)
    return filtered.astype(np.float32) / 255.0

def histogram_equalization(image):
    """
    Applies global Histogram Equalization.
    """
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_u8)
    return equalized.astype(np.float32) / 255.0

def contrast_enhancement(image, low_p=2, high_p=98):
    """
    Stretches contrast linearly between the low and high percentiles of active brain pixels.
    """
    active_pixels = image[image > 0]
    if len(active_pixels) == 0:
        return image
    
    v_min, v_max = np.percentile(active_pixels, (low_p, high_p))
    if v_min == v_max:
        return image
        
    enhanced = (image - v_min) / (v_max - v_min)
    enhanced = np.clip(enhanced, 0, 1)
    enhanced[image == 0] = 0.0
    return enhanced

def register_images(moving_image, reference_image):
    """
    Registers a moving image to a reference image using ORB keypoint matching and RANSAC.
    """
    moving_u8 = (np.clip(moving_image, 0, 1) * 255).astype(np.uint8)
    reference_u8 = (np.clip(reference_image, 0, 1) * 255).astype(np.uint8)
    
    # 1. Detect ORB features
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(moving_u8, None)
    kp2, des2 = orb.detectAndCompute(reference_u8, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return moving_image, np.eye(3) # Return original if feature matching fails
        
    # 2. Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Use top 50 matches
    good_matches = matches[:50]
    
    if len(good_matches) < 4:
        return moving_image, np.eye(3)
        
    # 3. Extract keypoint positions
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 4. Calculate Homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return moving_image, np.eye(3)
        
    # 5. Warp moving image
    h, w = reference_image.shape
    registered_image = cv2.warpPerspective(moving_image, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return registered_image, H

def augment_data(image, mask, rot_range=15, shift_range=10, scale_range=0.1, flip=True):
    """
    Performs data augmentation on image and mask pairs.
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    
    # Random parameters
    angle = np.random.uniform(-rot_range, rot_range)
    dx = np.random.uniform(-shift_range, shift_range)
    dy = np.random.uniform(-shift_range, shift_range)
    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    
    # Affine transformation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    
    aug_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aug_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Random horizontal flip
    if flip and np.random.rand() > 0.5:
        aug_image = cv2.flip(aug_image, 1)
        aug_mask = cv2.flip(aug_mask, 1)
        
    return aug_image, aug_mask

def run_preprocessing_pipeline(image, steps):
    """
    Runs a list of preprocessing steps sequentially.
    steps: list of strings (e.g., ['strip', 'noise', 'clahe', 'norm'])
    """
    current_img = image.copy()
    history = {'original': image.copy()}
    
    for step in steps:
        if step == 'strip':
            current_img, _ = skull_strip(current_img)
        elif step == 'norm':
            current_img = zscore_normalize(current_img)
        elif step == 'clahe':
            current_img = apply_clahe(current_img)
        elif step == 'noise':
            current_img = reduce_noise_bilateral(current_img)
        elif step == 'histeq':
            current_img = histogram_equalization(current_img)
        elif step == 'contrast':
            current_img = contrast_enhancement(current_img)
        history[step] = current_img.copy()
        
    return current_img, history
