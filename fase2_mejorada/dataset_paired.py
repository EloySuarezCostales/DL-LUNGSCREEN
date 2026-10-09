import os
import io
import zipfile
from typing import Literal, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Lista completa de las 14 patologías del dataset CheXpert.
CHEXPERT_TASKS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

class PairedCheXpertDataset(Dataset):
    """
    Dataset de PyTorch optimizado para el entrenamiento bimodal (Dual-Stream).
    Lee un CSV que ya contiene pares (Frontal, Lateral) pre-filtrados.
    """
    
    def __init__(
        self,
        csv_path: str,
        zip_path: str,
        uncertainty_policy: Literal['U-Ones', 'U-Zeroes', 'U-Ignore'] = 'U-Ignore',
        transform_frontal: Optional[transforms.Compose] = None,
        transform_lateral: Optional[transforms.Compose] = None,
        target_columns: List[str] = CHEXPERT_TASKS
    ) -> None:
        super().__init__()
        
        self.zip_path = zip_path
        self.uncertainty_policy = uncertainty_policy
        self.transform_frontal = transform_frontal
        self.transform_lateral = transform_lateral
        self.target_columns = target_columns
        
        # Puntero al archivo ZIP
        self.zip_file = None
        
        # 1. Cargar el CSV ya pareado (O(1) para el DataLoader)
        df = pd.read_csv(csv_path)
        
        # 2. Corregir las rutas (Kaggle elimina la carpeta 'CheXpert-v1.0-small/')
        df['Path_Frontal'] = df['Path_Frontal'].str.replace('CheXpert-v1.0-small/', '')
        df['Path_Lateral'] = df['Path_Lateral'].str.replace('CheXpert-v1.0-small/', '')
        
        # 3. Guardar las rutas en memoria
        self.paths_frontal = df['Path_Frontal'].values
        self.paths_lateral = df['Path_Lateral'].values
        
        # 4. Extracción de etiquetas: los valores vacíos se asumen como 0.0
        labels_raw = df[self.target_columns].fillna(0.0).values
        
        # 5. Aplicar la política de incertidumbre
        self.labels = self._apply_uncertainty_policy(labels_raw)

    def _apply_uncertainty_policy(self, labels: np.ndarray) -> torch.Tensor:
        labels_processed = labels.copy()
        
        if self.uncertainty_policy == 'U-Ones':
            labels_processed[labels_processed == -1.0] = 1.0
        elif self.uncertainty_policy == 'U-Zeroes':
            labels_processed[labels_processed == -1.0] = 0.0
        elif self.uncertainty_policy == 'U-Ignore':
            pass
        else:
            raise ValueError(f"Política de incertidumbre inválida: {self.uncertainty_policy}")
            
        return torch.FloatTensor(labels_processed)

    def __len__(self) -> int:
        return len(self.paths_frontal)

    def _load_image(self, path: str) -> Image.Image:
        """Función auxiliar para leer la imagen desde ZIP o directorio local."""
        if self.zip_path.endswith('.zip'):
            if self.zip_file is None:
                self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
                
            img_internal_path = path.replace('\\', '/')
            try:
                img_bytes = self.zip_file.read(img_internal_path)
                return Image.open(io.BytesIO(img_bytes)).convert('L')
            except Exception as e:
                raise IOError(f"Error al abrir la imagen {img_internal_path} desde el ZIP: {e}")
        else:
            img_internal_path_os = os.path.normpath(path)
            full_path = os.path.join(self.zip_path, img_internal_path_os)
            try:
                return Image.open(full_path).convert('L')
            except Exception as e:
                raise IOError(f"Error al abrir la imagen en {full_path}: {e}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna la tupla: (tensor_frontal, tensor_lateral, etiquetas)
        """
        # Cargar imágenes crudas
        img_front = self._load_image(self.paths_frontal[idx])
        img_lat = self._load_image(self.paths_lateral[idx])
        
        # Aplicar Data Augmentation asíncrono
        if self.transform_frontal:
            tensor_front = self.transform_frontal(img_front)
        else:
            tensor_front = transforms.ToTensor()(img_front)
            
        if self.transform_lateral:
            tensor_lat = self.transform_lateral(img_lat)
        else:
            tensor_lat = transforms.ToTensor()(img_lat)
            
        label = self.labels[idx]
        
        return tensor_front, tensor_lat, label
