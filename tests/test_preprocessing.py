import numpy as np
from src.preprocessing import skull_strip, zscore_normalize, apply_clahe, reduce_noise_bilateral

def test_zscore_normalize():
    # Create random image
    img = np.random.rand(128, 128)
    img[0:10, 0:10] = 0.0 # Set some zero background
    
    normalized = zscore_normalize(img)
    
    # Assert background is still zero
    assert np.all(normalized[img == 0] == 0.0)
    
    # Assert active pixels have mean ~0 and std ~1
    active = normalized[img > 0]
    assert np.allclose(np.mean(active), 0.0, atol=1e-5)
    assert np.allclose(np.std(active), 1.0, atol=1e-5)

def test_apply_clahe():
    img = np.random.rand(128, 128)
    enhanced = apply_clahe(img)
    
    assert enhanced.shape == (128, 128)
    assert enhanced.min() >= 0.0
    assert enhanced.max() <= 1.0

def test_reduce_noise_bilateral():
    img = np.random.rand(128, 128)
    filtered = reduce_noise_bilateral(img)
    
    assert filtered.shape == (128, 128)
    assert filtered.min() >= 0.0
    assert filtered.max() <= 1.0

def test_skull_strip():
    # Generate mock head (ellipse) with noise
    img = np.zeros((128, 128), dtype=np.float32)
    cv2_ellipse = np.zeros((128, 128), dtype=np.uint8)
    import cv2
    cv2.ellipse(cv2_ellipse, (64, 64), (40, 50), 0, 0, 360, 255, -1)
    
    # Parenchyma inside skull
    img[cv2_ellipse > 0] = 0.6
    
    # Skull stripped
    stripped, mask = skull_strip(img)
    
    assert stripped.shape == (128, 128)
    assert mask.shape == (128, 128)
    # The background corners must be stripped
    assert stripped[0, 0] == 0.0
    assert stripped[127, 127] == 0.0
