import numpy as np
import torch
from src.segmentation import (
    UNet, AttentionUNet, UNetPlusPlus, MaskRCNNWrapper,
    compute_dice, compute_iou, compute_precision, compute_recall
)

def test_models_forward_shape():
    # Verify model forward passes with CPU tensors
    dummy_input = torch.randn(1, 1, 64, 64) # Small size for speed
    
    unet = UNet(in_channels=1, out_channels=1)
    att_unet = AttentionUNet(in_channels=1, out_channels=1)
    unetpp = UNetPlusPlus(in_channels=1, out_channels=1)
    maskrcnn = MaskRCNNWrapper(in_channels=1)
    
    unet.eval()
    att_unet.eval()
    unetpp.eval()
    maskrcnn.eval()
    
    with torch.no_grad():
        out_unet = unet(dummy_input)
        out_att = att_unet(dummy_input)
        out_pp = unetpp(dummy_input)
        out_mask, out_box = maskrcnn(dummy_input)
        
    assert out_unet.shape == (1, 1, 64, 64)
    assert out_att.shape == (1, 1, 64, 64)
    assert out_pp.shape == (1, 1, 64, 64)
    assert out_mask.shape == (1, 1, 64, 64)
    assert out_box.shape == (1, 5) # [x, y, w, h, class_score]

def test_segmentation_metrics():
    # Create simple mock masks (circle)
    gt = np.zeros((100, 100), dtype=np.float32)
    pred = np.zeros((100, 100), dtype=np.float32)
    
    # Square region for GT
    gt[30:70, 30:70] = 1.0
    # Overlapping region for Pred
    pred[35:75, 35:75] = 1.0
    
    # Calculations
    intersection = np.sum(gt * pred) # 35*35 = 1225
    union = np.sum(gt) + np.sum(pred) - intersection # 1600 + 1600 - 1225 = 1975
    
    dice_expected = 2.0 * intersection / (np.sum(gt) + np.sum(pred)) # 2450 / 3200 = 0.7656
    iou_expected = intersection / union # 1225 / 1975 = 0.6202
    
    dice = compute_dice(pred, gt)
    iou = compute_iou(pred, gt)
    precision = compute_precision(pred, gt)
    recall = compute_recall(pred, gt)
    
    assert np.allclose(dice, dice_expected)
    assert np.allclose(iou, iou_expected)
    assert precision == intersection / np.sum(pred)
    assert recall == intersection / np.sum(gt)
