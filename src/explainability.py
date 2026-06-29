import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# =====================================================================
# 1. Gradient-based & Activation-based Explanations
# =====================================================================

def find_last_conv_layer(model):
    """
    Traverses the model backwards to find the last 2D convolutional layer.
    """
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in the model.")

class CAMExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.target_layer = find_last_conv_layer(model)
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output
        
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _reset(self):
        self.activations = None
        self.gradients = None

    def run_grad_cam(self, input_tensor, target_class=1):
        self._reset()
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass
        loss = output[0, target_class]
        loss.backward(retain_graph=True)
        
        # Calculate Grad-CAM
        grads = self.gradients.cpu().data.numpy()[0] # (C, H_f, W_f)
        acts = self.activations.cpu().data.numpy()[0] # (C, H_f, W_f)
        
        # Global average pooling of gradients
        weights = np.mean(grads, axis=(1, 2)) # (C,)
        
        # Weighted combination
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = np.maximum(cam, 0) # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # Resize to input dimensions
        h_in, w_in = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w_in, h_in))
        return cam

    def run_grad_cam_plusplus(self, input_tensor, target_class=1):
        self._reset()
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        loss = output[0, target_class]
        loss.backward(retain_graph=True)
        
        grads = self.gradients.cpu().data.numpy()[0]
        acts = self.activations.cpu().data.numpy()[0]
        
        # Grad-CAM++ calculation
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        
        sum_activations = np.sum(acts, axis=(1, 2))
        eps = 1e-7
        
        # Calculate alpha weights
        aij = np.zeros_like(grads)
        for c in range(grads.shape[0]):
            numerator = grads_power_2[c]
            denominator = 2 * grads_power_2[c] + sum_activations[c] * grads_power_3[c]
            denominator = np.where(denominator != 0, denominator, denominator + eps)
            aij[c] = numerator / denominator
            
        # Positive gradients
        pos_grads = np.maximum(grads, 0)
        weights = np.sum(aij * pos_grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
            
        h_in, w_in = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w_in, h_in))
        return cam

    def run_score_cam(self, input_tensor, target_class=1):
        """
        Score-CAM: Gradient-free CAM using forward activations.
        """
        self._reset()
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        acts = self.activations.clone() # (1, C, H_f, W_f)
        _, c_dim, h_f, w_f = acts.shape
        h_in, w_in = input_tensor.shape[2:]
        
        cam = torch.zeros((h_in, w_in), device=input_tensor.device)
        
        # Mask input with upsampled activations and compute target score increases
        for i in range(c_dim):
            act_map = acts[0, i, :, :].unsqueeze(0).unsqueeze(0) # (1, 1, H_f, W_f)
            # Upsample mask to input shape
            act_map_upsampled = F.interpolate(act_map, size=(h_in, w_in), mode='bilinear', align_corners=False)
            
            # Normalize map
            map_max = act_map_upsampled.max()
            map_min = act_map_upsampled.min()
            if map_max > map_min:
                act_map_upsampled = (act_map_upsampled - map_min) / (map_max - map_min)
                
            # Mask input
            masked_input = input_tensor * act_map_upsampled
            
            # Run prediction
            with torch.no_grad():
                output = self.model(masked_input)
                score = torch.softmax(output, dim=1)[0, target_class]
                
            cam += score * act_map_upsampled[0, 0]
            
        cam_np = cam.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        if cam_np.max() > 0:
            cam_np /= cam_np.max()
            
        return cam_np

# =====================================================================
# 2. Integrated Gradients
# =====================================================================

def run_integrated_gradients(model, input_tensor, target_class=1, steps=20):
    """
    Calculates Integrated Gradients relative to a black baseline image.
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor)
    
    # Generate scaled inputs along path: baseline + alpha * (input - baseline)
    scaled_inputs = []
    for step in range(steps + 1):
        alpha = step / steps
        scaled = baseline + alpha * (input_tensor - baseline)
        scaled_inputs.append(scaled)
        
    # Stack inputs
    scaled_inputs = torch.cat(scaled_inputs, dim=0).requires_grad_(True)
    
    # Forward pass
    outputs = model(scaled_inputs)
    scores = outputs[:, target_class]
    
    # Backward pass to get gradients
    grads = torch.autograd.grad(scores.sum(), scaled_inputs)[0] # (steps+1, 1, H, W)
    
    # Average gradients (trapezoidal rule approximation)
    avg_grads = torch.mean(grads, dim=0) # (1, H, W)
    
    # Multiply by difference (input - baseline)
    ig = (input_tensor - baseline) * avg_grads
    ig = ig.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    # Take absolute value / positive contributions
    ig = np.maximum(ig, 0)
    if ig.max() > 0:
        ig = ig / ig.max()
        
    return ig

# =====================================================================
# 3. Superpixel-based Perturbation SHAP
# =====================================================================

def run_shap_summary(model, image, target_class=1, num_samples=100):
    """
    A fast, superpixel perturbation-based explainer that approximates SHAP values
    on an 8x8 grid of blocks.
    """
    model.eval()
    h, w = image.shape
    grid_size = 8
    block_h, block_w = h // grid_size, w // grid_size
    
    # Create mask matrix
    # Samples: binary array indicating whether grid cell is active (1) or masked (0)
    samples = np.random.binomial(1, 0.5, size=(num_samples, grid_size * grid_size))
    # Make sure base image (all ones) and baseline (all zeros) are evaluated
    samples[0, :] = 1
    samples[1, :] = 0
    
    # Get base probability (with fully active image)
    tensor_in = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        base_logits = model(tensor_in)
        base_prob = torch.softmax(base_logits, dim=1)[0, target_class].item()
        
    # Evaluate perturbed samples
    y = []
    for s in samples:
        # Create perturbed image
        p_img = image.copy()
        for idx in range(len(s)):
            if s[idx] == 0:
                # Mask out this block
                r = idx // grid_size
                c = idx % grid_size
                p_img[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w] = 0.0
                
        p_tensor = torch.tensor(p_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(p_tensor)
            prob = torch.softmax(logits, dim=1)[0, target_class].item()
        y.append(prob)
        
    y = np.array(y)
    
    # Linear Regression (KernelSHAP simplified: fit weights directly)
    # Solve linear system: Samples * ShapleyValues = OutputChanges
    # We add L2 regularization (Ridge) for stability
    X = samples - 0.5 # Center features
    reg = 0.1
    # Ridge solver: W = (X^T * X + reg * I)^-1 * X^T * Y
    w_coef = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ (y - np.mean(y)))
    
    # Rescale weights to [0, 1] range for visual heatmap
    shap_val = w_coef.reshape((grid_size, grid_size))
    shap_val = np.maximum(shap_val, 0) # Focus on positive contributions
    
    # Upsample to image size
    shap_map = cv2.resize(shap_val, (w, h), interpolation=cv2.INTER_NEAREST)
    if shap_map.max() > 0:
        shap_map = shap_map / shap_map.max()
        
    return shap_map

# =====================================================================
# 4. Focus Quality & Confidence Scoring
# =====================================================================

def evaluate_focus_quality(heatmap, mask, threshold=0.4):
    """
    Evaluates where the model is focusing relative to the segmented tumor.
    Returns:
        overlap (float): IoU overlap between heatmap and mask
        category (str): 'Correct Focus', 'Partially Correct Focus', or 'Incorrect Focus'
    """
    h_bin = heatmap > threshold
    m_bin = mask > 0.5
    
    intersection = np.sum(h_bin & m_bin)
    union = np.sum(h_bin | m_bin)
    
    if union == 0:
        return 1.0, "Correct Focus"
        
    overlap = intersection / union
    
    if overlap >= 0.45:
        category = "Correct Focus"
    elif overlap >= 0.15:
        category = "Partially Correct Focus"
    else:
        category = "Incorrect Focus"
        
    return float(overlap), category

def compute_explainability_confidence(overlap, prediction_prob):
    """
    Computes an explainability-informed confidence score.
    High confidence only when the model predicts with high probability AND focuses on the correct area.
    """
    return float(0.4 * overlap + 0.6 * prediction_prob)

def get_xai_visualization(image, heatmap, title="Grad-CAM"):
    """
    Generates a color overlay of the XAI heatmap onto the grayscale MRI.
    Uses JET colormap for hot spot visibility.
    """
    if image is None or heatmap is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)
        
    image = np.asarray(image, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    
    # 1. Normalize image safely to [0, 1] range
    img_min, img_max = np.min(image), np.max(image)
    if img_max > img_min:
        image_norm = (image - img_min) / (img_max - img_min)
    else:
        image_norm = np.zeros_like(image)
        
    # 2. Normalize heatmap safely to [0, 1] range
    hm_min, hm_max = np.min(heatmap), np.max(heatmap)
    if hm_max > hm_min:
        heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)
        
    # 3. Handle channel dims for resizing and grayscale matching
    if len(image_norm.shape) == 3:
        if image_norm.shape[2] == 1:
            image_gray = image_norm[:, :, 0]
        elif image_norm.shape[2] == 3:
            image_gray = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_norm[:, :, 0]
    else:
        image_gray = image_norm
        
    # Resize heatmap to match grayscale image shape
    if heatmap_norm.shape != image_gray.shape:
        heatmap_norm = cv2.resize(heatmap_norm, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    # Scale to uint8
    img_u8 = (image_gray * 255).astype(np.uint8)
    heatmap_u8 = (heatmap_norm * 255).astype(np.uint8)
    
    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    
    # Convert image to 3-channel BGR
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
    # Overlay: 60% image, 40% heatmap
    overlay = cv2.addWeighted(img_bgr, 0.6, color_heatmap, 0.4, 0)
    
    return overlay
