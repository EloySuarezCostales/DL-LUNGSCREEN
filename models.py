import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class CheXpertDenseNet(nn.Module):
    """
    Modelo DenseNet-121 adaptado para la clasificación de patologías de CheXpert (Fase 1).
    - Modifica la primera capa (conv0) para aceptar imágenes de 1 canal (escala de grises).
    - Inicializa los pesos con ImageNet (Transfer Learning).
    - Modifica el clasificador final para el número de patologías objetivo.
    - Permite extraer el vector latente de 1024 dimensiones para la Fase 2 (Late Fusion).
    """
    
    def __init__(self, num_classes: int = 14):
        super().__init__()
        
        # 1. Cargar el backbone preentrenado con ImageNet (Regla estricta del AGENTS.md)
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # 2. Adaptar la primera capa convolucional (conv0) de 3 canales (RGB) a 1 canal (Grises)
        # La conv0 original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv0 = self.backbone.features.conv0
        
        new_conv0 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv0.out_channels, 
            kernel_size=original_conv0.kernel_size, 
            stride=original_conv0.stride, 
            padding=original_conv0.padding, 
            bias=False
        )
        
        # Transfer Learning: Para no perder lo aprendido en conv0 con ImageNet, calculamos el promedio 
        # de los pesos de los 3 canales de color originales. Esto es lo que estipula el AGENTS.md.
        with torch.no_grad():
            new_conv0.weight = nn.Parameter(torch.mean(original_conv0.weight, dim=1, keepdim=True))
            
        # Reemplazamos la capa en la arquitectura
        self.backbone.features.conv0 = new_conv0
        
        # 3. Adaptar el clasificador final
        # DenseNet121 genera un vector latente de 1024 características.
        self.num_features = self.backbone.classifier.in_features # Devuelve 1024
        
        # Reemplazamos la capa final para que emita 'num_classes' predicciones.
        self.backbone.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x: torch.Tensor, extract_features: bool = False) -> torch.Tensor:
        """
        Paso hacia adelante (Forward pass) de la red.
        
        Args:
            x (torch.Tensor): Tensor de imágenes de entrada, tamaño: (Batch, 1, Altura, Anchura).
            extract_features (bool): Si es True, detiene el paso antes del clasificador y 
                                     devuelve el vector latente de 1024 dimensiones. 
                                     ¡Esto será clave para la Fase 2 (Late Fusion)!
        
        Returns:
            torch.Tensor: Logits de clasificación (B, num_classes) o Vector latente (B, 1024).
        """
        # 1. Extracción de mapas de características (Feature maps)
        features = self.backbone.features(x)
        
        # 2. Operaciones finales estándar de DenseNet (ReLU + Pooling)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1) # Nos queda el vector de (Batch_Size, 1024)
        
        # ¿Estamos en la Fase 2 y solo queremos las características para concatenarlas?
        if extract_features:
            return out
            
        # Fase 1: Pasamos el vector por el clasificador para obtener las predicciones
        out = self.backbone.classifier(out)
        return out
