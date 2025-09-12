"""
U-Net++ segmentation model with ResNet encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):
    """U-Net++ with ResNet encoder for prostate tissue segmentation"""
    
    def __init__(self, num_classes=4, encoder_name="resnet34", pretrained=True):
        super(SegmentationModel, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        
    def forward(self, x):
        return self.model(x)

class CombinedLoss(nn.Module):
    """Combined Dice + Cross Entropy Loss for handling class imbalance"""
    
    def __init__(self, weight_ce=1.0, weight_dice=1.0, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()
        
    def dice_loss(self, pred, target, num_classes):
        """Calculate Dice loss for multi-class segmentation"""
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        return 1 - torch.stack(dice_scores).mean()
    
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target, pred.shape[1])
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
