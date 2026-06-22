import torch
import cv2
import numpy as np
from src.classification import get_classification_model

def crop_roi(image, mask, padding=15):
    """
    Extracts the region of interest (ROI) from the MRI image based on the segmentation mask.
    Pads the bounding box to capture surrounding tissue context (important for edge margins).
    
    Returns:
        roi (np.ndarray): Cropped and resized ROI (224x224)
        bbox (tuple): (x_min, y_min, x_max, y_max)
    """
    h, w = image.shape
    pts = np.argwhere(mask > 0.5)
    
    if len(pts) == 0:
        # Fallback to central region if no tumor is detected
        x_min, y_min = w // 4, h // 4
        x_max, y_max = 3 * w // 4, 3 * h // 4
    else:
        # Bounding box coordinates
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        
        # Apply padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
    # Crop
    cropped = image[y_min:y_max, x_min:x_max]
    
    # Resize to classification input dimensions (224x224)
    roi = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    return roi, (x_min, y_min, x_max, y_max)

def run_pipeline_a(image, model_name="ResNet50"):
    """
    Pipeline A: Whole MRI image -> Classifier.
    """
    model = get_classification_model(model_name)
    model.eval()
    
    # Resize whole MRI to 224x224
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor
    tensor_in = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 224, 224)
    
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
    return probs, img_resized

def run_pipeline_b(image, mask, model_name="ResNet50"):
    """
    Pipeline B: MRI -> Segmentation -> ROI Crop -> Classifier.
    """
    model = get_classification_model(model_name)
    model.eval()
    
    # Crop ROI
    roi, bbox = crop_roi(image, mask)
    
    # Convert to tensor
    tensor_in = torch.tensor(roi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
    return probs, roi, bbox

def run_pipeline_c(image, mask, cnn_name="ResNet50", vit_name="Vision Transformer"):
    """
    Pipeline C: MRI -> Segmentation -> ROI Crop -> CNN + ViT Ensemble.
    Averages the probabilities of a convolutional network and a transformer.
    """
    cnn_model = get_classification_model(cnn_name)
    vit_model = get_classification_model(vit_name)
    
    cnn_model.eval()
    vit_model.eval()
    
    # Crop ROI
    roi, bbox = crop_roi(image, mask)
    
    # Convert to tensor
    tensor_in = torch.tensor(roi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        cnn_logits = cnn_model(tensor_in)
        vit_logits = vit_model(tensor_in)
        
        cnn_probs = torch.softmax(cnn_logits, dim=1).numpy()[0]
        vit_probs = torch.softmax(vit_logits, dim=1).numpy()[0]
        
    # Ensemble probability: Soft voting (mean)
    ensemble_probs = (cnn_probs + vit_probs) / 2.0
    
    return ensemble_probs, roi, bbox, cnn_probs, vit_probs
