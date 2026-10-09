import torch
import torch.nn as nn
import sys
import os

from models import CheXpertDenseNet

class CrossAttentionFusion(nn.Module):
    """
    Mecanismo de Fusión SOTA basado en Multi-Head Cross-Attention.
    Permite que la vista Frontal consulte proactivamente las características de la Lateral (y viceversa).
    """
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, num_classes: int = 14):
        super().__init__()
        # Atención cruzada direccional (Frontal pregunta a Lateral, y Lateral pregunta a Frontal)
        self.attn_f2l = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_l2f = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Normalización para estabilizar el entrenamiento
        self.norm_f = nn.LayerNorm(embed_dim)
        self.norm_l = nn.LayerNorm(embed_dim)
        
        # MLP Final que procesa los 2048 parámetros enriquecidos por la atención
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, z_frontal: torch.Tensor, z_lateral: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention en PyTorch con batch_first=True espera (Batch, Seq_len, Features)
        # Como solo tenemos 1 vector global por imagen, Seq_len = 1
        q_f = z_frontal.unsqueeze(1)
        kv_l = z_lateral.unsqueeze(1)
        
        q_l = z_lateral.unsqueeze(1)
        kv_f = z_frontal.unsqueeze(1)
        
        # Frontal consulta Lateral
        attn_f2l_out, _ = self.attn_f2l(q_f, kv_l, kv_l)
        f_enhanced = self.norm_f(q_f + attn_f2l_out).squeeze(1) # Suma residual + Norm
        
        # Lateral consulta Frontal
        attn_l2f_out, _ = self.attn_l2f(q_l, kv_f, kv_f)
        l_enhanced = self.norm_l(q_l + attn_l2f_out).squeeze(1) # Suma residual + Norm
        
        # Concatenación de vectores enriquecidos (1024 + 1024 = 2048)
        z_fusion = torch.cat([f_enhanced, l_enhanced], dim=1)
        
        # Emisión de los 14 logits finales
        return self.mlp(z_fusion)


class DualStreamDenseNet(nn.Module):
    """
    Arquitectura Bimodal SOTA para DL-LUNGSCREENING (Fase 2).
    - Carga los extractores pre-entrenados de la Fase 1.
    - Utiliza Cross-Attention en lugar de concatenación plana.
    - Permite descongelación gradual (Gradual Unfreezing) en el entrenamiento.
    """
    
    def __init__(
        self, 
        num_classes: int = 14, 
        frontal_checkpoint_path: str = None, 
        lateral_checkpoint_path: str = None,
        freeze_backbones: bool = True
    ):
        super().__init__()
        
        # 1. Instanciar los "Ojos"
        self.frontal_extractor = CheXpertDenseNet(num_classes=num_classes)
        self.lateral_extractor = CheXpertDenseNet(num_classes=num_classes)
        
        # 2. Cargar los pesos de la Fase 1
        if frontal_checkpoint_path:
            self._load_pretrained_weights(self.frontal_extractor, frontal_checkpoint_path)
        if lateral_checkpoint_path:
            self._load_pretrained_weights(self.lateral_extractor, lateral_checkpoint_path)
            
        # 3. Congelación Inicial (Warm-up phase)
        if freeze_backbones:
            for param in self.frontal_extractor.parameters():
                param.requires_grad = False
            for param in self.lateral_extractor.parameters():
                param.requires_grad = False
                
        # Bypass de las cabezas de clasificación originales
        self.frontal_extractor.backbone.classifier = nn.Identity()
        self.lateral_extractor.backbone.classifier = nn.Identity()

        # 4. Instanciar el "Cerebro" con Cross-Attention
        self.fusion_head = CrossAttentionFusion(embed_dim=1024, num_heads=8, num_classes=num_classes)

    def _load_pretrained_weights(self, model: nn.Module, checkpoint_path: str):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"[+] Pesos cargados correctamente desde {checkpoint_path}")
        except Exception as e:
            print(f"[!] Error al cargar el checkpoint {checkpoint_path}: {e}")
            raise e

    def unfreeze_backbones(self):
        """Función llamada desde train_dual.py para iniciar la etapa de Fine-Tuning."""
        for param in self.frontal_extractor.parameters():
            param.requires_grad = True
        for param in self.lateral_extractor.parameters():
            param.requires_grad = True
        print("[+] Ojos (Backbones) descongelados correctamente.")

    def forward(self, img_frontal: torch.Tensor, img_lateral: torch.Tensor) -> torch.Tensor:
        # Extraer vectores de 1024 dimensiones
        z_frontal = self.frontal_extractor(img_frontal, extract_features=True)
        z_lateral = self.lateral_extractor(img_lateral, extract_features=True)
        
        # Pasar por el Cerebro con Cross-Attention
        out = self.fusion_head(z_frontal, z_lateral)
        return out
