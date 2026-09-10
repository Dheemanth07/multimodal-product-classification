"""
Multimodal Product Classification - Model Architecture
======================================================
Dual-Branch Deep Neural Network fusing Vision and NLP representations:
1. Vision Branch: MobileNetV2 (1280 features) or ResNet-18 (512 features)
2. NLP Branch: Word Embedding + 2-layer Bidirectional GRU + Temporal Pooling (256 features)
3. Fusion Layer: Manual Concatenation + Multi-Layer Perceptron (MLP) Classifier

Strictly adheres to:
- Parameter Cap: < 15,000,000 parameters (strictly verified by assertion)
- No VLMs: Built manually using standard CNN and RNN primitives
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisionBranch(nn.Module):
    """
    Vision feature extractor using lightweight pre-trained backbones (Module 3).
    Supports MobileNetV2 (default) and ResNet-18.
    """
    def __init__(self, backbone_name="mobilenet_v2", pretrained=True):
        super(VisionBranch, self).__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            try:
                base_model = models.mobilenet_v2(weights=weights)
            except Exception:
                # Fallback if offline/local cache unavailable
                base_model = models.mobilenet_v2(weights=None)
            
            # Extract features only (without final 1000-class classifier)
            self.feature_extractor = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.out_features = 1280

        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                base_model = models.resnet18(weights=weights)
            except Exception:
                base_model = models.resnet18(weights=None)
            
            # Exclude final fully connected layer
            self.feature_extractor = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.out_features = 512

        elif self.backbone_name == "custom_cnn":
            # 4-stage convolutional backbone from scratch
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64x64

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32x32

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16x16

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.pool = nn.Identity()
            self.flatten = nn.Flatten()
            self.out_features = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        # x: [Batch_Size, 3, H, W]
        features = self.feature_extractor(x)  # [B, Channels, H', W']
        pooled = self.pool(features)          # [B, Channels, 1, 1]
        out = self.flatten(pooled)            # [B, out_features]
        return out


class TextBranch(nn.Module):
    """
    Sequential NLP feature extractor using trainable word embeddings
    and a Bidirectional GRU with Temporal Pooling (Modules 4 & 5).
    """
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(TextBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional GRU outputs 2 * hidden_dim
        self.out_features = hidden_dim * 2

    def forward(self, text_indices):
        # text_indices: [Batch_Size, Sequence_Length]
        # 1. Embed tokens
        embeds = self.embedding(text_indices)      # [B, L, embedding_dim]
        embeds = self.dropout(embeds)
        
        # 2. Sequential GRU forward
        gru_out, _ = self.gru(embeds)               # [B, L, 2 * hidden_dim]
        
        # 3. Mask-aware Temporal Pooling (ignore <PAD> tokens at index 0)
        mask = (text_indices != 0).unsqueeze(-1).float()  # [B, L, 1]
        masked_out = gru_out * mask
        sum_pooled = masked_out.sum(dim=1)                # [B, 2 * hidden_dim]
        lengths = mask.sum(dim=1).clamp(min=1.0)          # [B, 1]
        avg_pooled = sum_pooled / lengths                 # [B, 2 * hidden_dim]
        
        return avg_pooled


class DualBranchMultimodalModel(nn.Module):
    """
    End-to-End Dual-Branch Multimodal Network:
    - Vision: Lightweight Backbone (MobileNetV2 / ResNet-18)
    - NLP: Bidirectional GRU
    - Fusion: Tensor Concatenation
    - Classification: Multi-Layer Perceptron (MLP)
    """
    def __init__(self, 
                 num_classes=140, 
                 vocab_size=10000, 
                 vision_backbone="mobilenet_v2", 
                 pretrained_vision=True,
                 embed_dim=128, 
                 hidden_dim=128,
                 num_gru_layers=2,
                 fusion_hidden_dim=512,
                 dropout_rate=0.3):
        super(DualBranchMultimodalModel, self).__init__()
        
        # 1. Vision Branch
        self.vision_branch = VisionBranch(backbone_name=vision_backbone, pretrained=pretrained_vision)
        vision_feat_dim = self.vision_branch.out_features
        
        # 2. NLP Branch
        self.nlp_branch = TextBranch(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout_rate
        )
        nlp_feat_dim = self.nlp_branch.out_features
        
        # 3. Manual Concatenation Fusion Layer
        fused_dim = vision_feat_dim + nlp_feat_dim
        
        # 4. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, images, texts):
        """
        Forward pass tracing:
        - images: [B, 3, H, W] -> img_feat: [B, D_vision]
        - texts:  [B, L]       -> text_feat: [B, D_text]
        - fused:  [B, D_vision + D_text]
        - logits: [B, num_classes]
        """
        # Modality 1: Image features
        img_feat = self.vision_branch(images)
        
        # Modality 2: Text features
        text_feat = self.nlp_branch(texts)
        
        # Late Fusion via Concatenation
        fused = torch.cat((img_feat, text_feat), dim=1)
        
        # Prediction Logits
        logits = self.classifier(fused)
        return logits


def get_model_summary_and_check_cap(model, max_param_cap=15_000_000):
    """
    Computes total parameter count, prints architectural breakdown,
    and strictly validates compliance with the 15M parameter limit.
    """
    vision_params = sum(p.numel() for p in model.vision_branch.parameters())
    nlp_params = sum(p.numel() for p in model.nlp_branch.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 65)
    print("      MULTIMODAL DUAL-BRANCH MODEL PARAMETER AUDIT")
    print("=" * 65)
    print(f"  * Vision Branch Parameters:       {vision_params:>12,}")
    print(f"  * NLP Branch Parameters:          {nlp_params:>12,}")
    print(f"  * Fusion Classifier Parameters:   {classifier_params:>12,}")
    print("-" * 65)
    print(f"  * Total Parameters (All):         {total_params:>12,}")
    print(f"  * Trainable Parameters:           {trainable_params:>12,}")
    print(f"  * Hard Parameter Cap Limit:       {max_param_cap:>12,}")
    remaining_budget = max_param_cap - total_params
    print(f"  * Remaining Parameter Headroom:   {remaining_budget:>12,}")
    print("=" * 65)
    
    assert total_params <= max_param_cap, (
        f"CRITICAL VIOLATION: Model has {total_params:,} parameters, "
        f"exceeding the strict limit of {max_param_cap:,}!"
    )
    print(" [PASSED] Model strictly complies with the < 15,000,000 parameter limit.\n")
    return total_params
