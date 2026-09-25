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
# NOTA: Asegúrate de que la salida final del modelo en models.py coincida con la cantidad de elementos activos aquí.
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

class CheXpertDataset(Dataset):
    """
    Dataset de PyTorch para cargar imágenes directamente desde el ZIP de CheXpert.
    """
    
    def __init__(
        self,
        csv_path: str,
        zip_path: str,
        view_type: Literal['Frontal', 'Lateral'],
        uncertainty_policy: Literal['U-Ones', 'U-Zeroes', 'U-Ignore'] = 'U-Ignore',
        transform: Optional[transforms.Compose] = None,
        target_columns: List[str] = CHEXPERT_TASKS
    ) -> None:
        super().__init__()
        
        self.zip_path = zip_path
        self.view_type = view_type
        self.uncertainty_policy = uncertainty_policy
        self.transform = transform
        self.target_columns = target_columns
        
        # El puntero al archivo ZIP se inicializa a None para evitar problemas 
        # al usar DataLoader con múltiples 'workers' (multiprocessing).
        self.zip_file = None
        
        # 1. Carga de metadatos (sirve tanto para train.csv como para valid.csv)
        df = pd.read_csv(csv_path)
        
        # 2. Filtrado estricto por tipo de vista (Requisito Fase 1)
        df = df[df['Frontal/Lateral'] == self.view_type].copy()
        
        # 3. Filtrado de seguridad: Evitar archivos ocultos/corruptos de macOS (._)
        df = df[~df['Path'].str.contains(r'/._', na=False, regex=True)].copy()
        
        # 4. Corregir la ruta del ZIP (Kaggle elimina la carpeta raíz 'CheXpert-v1.0-small/')
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '')
        
        self.image_paths = df['Path'].values
        
        # 4. Extracción de etiquetas: los valores vacíos se asumen como 0.0
        labels_raw = df[self.target_columns].fillna(0.0).values
        
        # 5. Aplicar la política de incertidumbre
        self.labels = self._apply_uncertainty_policy(labels_raw)

    def _apply_uncertainty_policy(self, labels: np.ndarray) -> torch.Tensor:
        """
        Aplica la política seleccionada sobre los valores inciertos (-1.0).
        """
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
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carga la imagen desde la RAM leyendo el archivo ZIP, fuerza 1 canal y retorna el tensor.
        """
        # Inicialización "Lazy" del archivo zip. Es obligatorio hacerlo dentro de __getitem__
        # si queremos que PyTorch pueda usar num_workers > 0 sin crashear.
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            
        img_internal_path = self.image_paths[idx]
        
        try:
            # Leemos los bytes directamente desde el ZIP (sin tocar el disco duro)
            img_bytes = self.zip_file.read(img_internal_path)
            # Convertimos esos bytes en una imagen y la forzamos a 1 canal (escala de grises)
            image = Image.open(io.BytesIO(img_bytes)).convert('L')
        except Exception as e:
            raise IOError(f"Error al abrir la imagen {img_internal_path} desde el ZIP: {e}")
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        label = self.labels[idx]
        
        return image, label
