"""
model.py — Stage 2
EfficientNet-B4 fine-tuned for 767-class Gardiner sign classification.

Architecture:
  EfficientNet-B4 (pretrained ImageNet)
    → Dropout(0.4)
    → Linear(1792, 512)
    → ReLU + BatchNorm
    → Dropout(0.3)
    → Linear(512, num_classes)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class HieroglyphClassifier(nn.Module):

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        # Load pretrained EfficientNet-B4
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Keep the feature extractor (everything except the classifier head)
        self.features = base.features
        self.avgpool  = base.avgpool

        # EfficientNet-B4 feature dim = 1792
        in_features = base.classifier[1].in_features  # 1792

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.75),
            nn.Linear(512, num_classes),
        )

        # Weight init for the new head
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze EfficientNet feature extractor — train head only (first epochs)."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Stage2] Backbone frozen — training head only.")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("[Stage2] Backbone unfrozen — full fine-tuning.")

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)