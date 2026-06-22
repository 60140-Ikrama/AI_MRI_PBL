import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

# =====================================================================
# 1. Architecutures
# =====================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(x3))
        
        # Decoder with Skip Connections
        d3 = self.up3(b)
        # Handle shape differences if any
        if d3.shape != x3.shape:
            d3 = F.interpolate(d3, size=x3.shape[2:])
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != x2.shape:
            d2 = F.interpolate(d2, size=x2.shape[2:])
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != x1.shape:
            d1 = F.interpolate(d1, size=x1.shape[2:])
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: number of channels in gating signal (coarser layer)
        F_l: number of channels in skip connection (finer layer)
        F_int: number of intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        # Upsample g1 if shape differs from x
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x1 = self.W_x(x)
        
        # Additive Attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        alpha = self.sigmoid(psi)
        
        return x * alpha

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=64, F_int=32)
        self.dec3 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=32, F_int=16)
        self.dec2 = DoubleConv(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=32, F_l=16, F_int=8)
        self.dec1 = DoubleConv(32, 16)
        
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        
        b = self.bottleneck(self.pool3(x3))
        
        d3 = self.up3(b)
        if d3.shape != x3.shape:
            d3 = F.interpolate(d3, size=x3.shape[2:])
        # Gating signal is the bottleneck output 'b', skip connection is x3
        x3_att = self.att3(b, x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != x2.shape:
            d2 = F.interpolate(d2, size=x2.shape[2:])
        x2_att = self.att2(d3, x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != x1.shape:
            d1 = F.interpolate(d1, size=x1.shape[2:])
        x1_att = self.att1(d2, x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        filters = [16, 32, 64, 128]
        
        self.enc0_0 = DoubleConv(in_channels, filters[0])
        self.enc1_0 = DoubleConv(filters[0], filters[1])
        self.enc2_0 = DoubleConv(filters[1], filters[2])
        self.enc3_0 = DoubleConv(filters[2], filters[3])
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Nested skip connections
        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2])
        
        self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1])
        
        self.conv0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0])
        
        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        x0_0 = self.enc0_0(x)
        x1_0 = self.enc1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.enc2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.enc3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        return torch.sigmoid(self.final(x0_3))

class MaskRCNNWrapper(nn.Module):
    """
    A custom PyTorch model representing a Mask R-CNN pipeline.
    It produces instance bounding boxes, class scores, and segmentation masks.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # Backbone (lightweight ResNet-like)
        self.backbone = nn.Sequential(
            DoubleConv(in_channels, 16),
            nn.MaxPool2d(2),
            DoubleConv(16, 32),
            nn.MaxPool2d(2)
        )
        # RPN & Box Head
        self.box_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4 + 1) # [x, y, w, h, class_score]
        )
        # Mask Head (Encoder-Decoder)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            DoubleConv(16, 16),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            DoubleConv(8, 8),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        box_out = self.box_head(feat)
        mask_out = torch.sigmoid(self.mask_head(feat))
        return mask_out, box_out

# =====================================================================
# 2. Evaluation Metrics (NumPy)
# =====================================================================

def compute_dice(pred, target, threshold=0.5):
    p = (pred > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)
    intersection = np.sum(p * t)
    total = np.sum(p) + np.sum(t)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total

def compute_iou(pred, target, threshold=0.5):
    p = (pred > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)
    intersection = np.sum(p * t)
    union = np.sum(p) + np.sum(t) - intersection
    if union == 0:
        return 1.0
    return intersection / union

def compute_precision(pred, target, threshold=0.5):
    p = (pred > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)
    intersection = np.sum(p * t)
    pred_total = np.sum(p)
    if pred_total == 0:
        return 1.0 if np.sum(t) == 0 else 0.0
    return intersection / pred_total

def compute_recall(pred, target, threshold=0.5):
    p = (pred > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)
    intersection = np.sum(p * t)
    target_total = np.sum(t)
    if target_total == 0:
        return 1.0
    return intersection / target_total

def compute_hausdorff(pred, target, threshold=0.5):
    p = (pred > threshold).astype(np.uint8)
    t = (target > threshold).astype(np.uint8)
    
    # Extract edge coordinates
    p_pts = np.argwhere(p > 0)
    t_pts = np.argwhere(t > 0)
    
    if len(p_pts) == 0 and len(t_pts) == 0:
        return 0.0
    if len(p_pts) == 0 or len(t_pts) == 0:
        return 256.0 # Max distance for 256x256 image
        
    d1 = directed_hausdorff(p_pts, t_pts)[0]
    d2 = directed_hausdorff(t_pts, p_pts)[0]
    return max(d1, d2)

def compute_boundary_f1(pred, target, threshold=0.5, d_threshold=3.0):
    """
    Computes Boundary F1 (BF-score) within a distance tolerance (d_threshold).
    """
    p = (pred > threshold).astype(np.uint8)
    t = (target > threshold).astype(np.uint8)
    
    # Get boundaries using Sobel filter or Canny
    p_edge = cv2.Canny(p * 255, 100, 200) > 0
    t_edge = cv2.Canny(t * 255, 100, 200) > 0
    
    p_pts = np.argwhere(p_edge)
    t_pts = np.argwhere(t_edge)
    
    if len(p_pts) == 0 and len(t_pts) == 0:
        return 1.0
    if len(p_pts) == 0 or len(t_pts) == 0:
        return 0.0
        
    # Compute distances from pred boundary points to target boundary points
    dists_p_to_t = []
    for pt in p_pts:
        dists = np.linalg.norm(t_pts - pt, axis=1)
        dists_p_to_t.append(np.min(dists))
        
    dists_t_to_p = []
    for pt in t_pts:
        dists = np.linalg.norm(p_pts - pt, axis=1)
        dists_t_to_p.append(np.min(dists))
        
    # Match within distance threshold
    precision = np.mean(np.array(dists_p_to_t) <= d_threshold)
    recall = np.mean(np.array(dists_t_to_p) <= d_threshold)
    
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def get_segmentation_metrics(pred, target, threshold=0.5):
    """
    Convenience function returning a dictionary of all segmentation metrics.
    """
    return {
        "Dice": compute_dice(pred, target, threshold),
        "IoU": compute_iou(pred, target, threshold),
        "Precision": compute_precision(pred, target, threshold),
        "Recall": compute_recall(pred, target, threshold),
        "Hausdorff": compute_hausdorff(pred, target, threshold),
        "Boundary F1": compute_boundary_f1(pred, target, threshold)
    }

def get_overlay_images(image, mask_gt, mask_pred):
    """
    Generates color overlays of ground truth and prediction on the MRI.
    Yellow = True Positive (Intersection)
    Red = False Positive (Over-segmentation)
    Blue = False Negative (Under-segmentation)
    Green = Ground Truth
    """
    h, w = image.shape
    # Scale image to 3-channel BGR [0, 255]
    img_bgr = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    overlay = img_bgr.copy()
    
    m_gt = mask_gt > 0.5
    m_pred = mask_pred > 0.5
    
    # 1. TP: Yellow
    tp = m_gt & m_pred
    overlay[tp] = [0, 255, 255] # Yellow in BGR
    
    # 2. FP: Red
    fp = (~m_gt) & m_pred
    overlay[fp] = [0, 0, 255] # Red in BGR
    
    # 3. FN: Blue
    fn = m_gt & (~m_pred)
    overlay[fn] = [255, 0, 0] # Blue in BGR
    
    # Blend overlay with original image
    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    
    # Error map: Red for FP, Blue for FN, green for TP
    error_map = np.zeros_like(img_bgr)
    error_map[tp] = [0, 255, 0] # Green
    error_map[fp] = [0, 0, 255] # Red
    error_map[fn] = [255, 0, 0] # Blue
    
    # Boundary visualization: green for GT boundary, red for pred boundary
    boundary_vis = img_bgr.copy()
    gt_edge = cv2.Canny((mask_gt * 255).astype(np.uint8), 100, 200) > 0
    pred_edge = cv2.Canny((mask_pred * 255).astype(np.uint8), 100, 200) > 0
    boundary_vis[gt_edge] = [0, 255, 0] # Green
    boundary_vis[pred_edge] = [0, 0, 255] # Red
    
    return blended, error_map, boundary_vis
