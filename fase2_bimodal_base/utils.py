import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Union, Literal

def calculate_pos_weights(
    csv_path: str,
    target_columns: List[str],
    view_type: Literal['Frontal', 'Lateral'],
    uncertainty_policy: Literal['U-Ones', 'U-Zeroes', 'U-Ignore'] = 'U-Zeroes'
) -> torch.Tensor:
    """
    Calcula los pesos positivos (pos_weight) para la función BCEWithLogitsLoss.
    Fórmula: pos_weight = N_negativos / N_positivos para cada clase.
    
    Args:
        csv_path (str): Ruta al archivo de entrenamiento (train.csv).
        target_columns (List[str]): Lista de las patologías a predecir.
        view_type (str): 'Frontal' o 'Lateral' para aislar las estadísticas a esa vista.
        uncertainty_policy (str): Cómo tratar las etiquetas -1.0 al contar los positivos.
        
    Returns:
        torch.Tensor: Tensor de pesos de tamaño (num_classes,)
    """
    df = pd.read_csv(csv_path)
    
    # 1. Filtrar por vista para que los pesos sean precisos al modelo que estamos entrenando
    df = df[df['Frontal/Lateral'] == view_type].copy()
    
    # 2. Extraer etiquetas puras (rellenando vacíos con 0.0)
    labels = df[target_columns].fillna(0.0).values
    
    # 3. Aplicar política de incertidumbre
    if uncertainty_policy == 'U-Ones':
        labels[labels == -1.0] = 1.0
    elif uncertainty_policy == 'U-Zeroes':
        labels[labels == -1.0] = 0.0
    elif uncertainty_policy == 'U-Ignore':
        # Para el cálculo de pesos, si ignoramos los -1.0, solo contamos 
        # estrictamente los que sabemos que son 1.0 o 0.0
        pass 
    
    pos_weights = []
    
    # Calcular peso por cada clase
    for i in range(labels.shape[1]):
        col_labels = labels[:, i]
        
        # Si usamos U-Ignore, filtramos los -1.0 antes de contar
        if uncertainty_policy == 'U-Ignore':
            col_labels = col_labels[col_labels != -1.0]
            
        n_positives = np.sum(col_labels == 1.0)
        n_negatives = np.sum(col_labels == 0.0)
        
        # Evitar división por cero si una clase no tiene positivos
        if n_positives == 0:
            weight = 1.0
        else:
            weight = n_negatives / n_positives
            
        pos_weights.append(weight)
        
    return torch.FloatTensor(pos_weights)


def compute_auroc(y_true: np.ndarray, y_pred: np.ndarray, target_columns: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calcula el AUROC macro y el AUROC individual de cada clase.
    
    Args:
        y_true (np.ndarray): Etiquetas reales (Ground Truth) de forma (N, num_classes).
        y_pred (np.ndarray): Predicciones del modelo (Probabilidades/Sigmoid) de forma (N, num_classes).
        target_columns (List[str]): Nombres de las clases.
        
    Returns:
        Dict: Contiene el 'macro_auroc' y un diccionario 'class_auroc' con los valores por clase.
    """
    auroc_scores = {}
    
    for i, class_name in enumerate(target_columns):
        true_labels = y_true[:, i]
        pred_probs = y_pred[:, i]
        
        # Filtrar valores -1.0 (En caso de que estemos evaluando sobre un subconjunto con U-Ignore)
        valid_mask = (true_labels != -1.0)
        
        if np.sum(valid_mask) > 0:
            true_valid = true_labels[valid_mask]
            pred_valid = pred_probs[valid_mask]
            
            # Solo podemos calcular ROC si hay al menos una muestra positiva y una negativa
            if len(np.unique(true_valid)) > 1:
                score = roc_auc_score(true_valid, pred_valid)
                auroc_scores[class_name] = score
            else:
                auroc_scores[class_name] = float('nan')
        else:
            auroc_scores[class_name] = float('nan')
            
    # Calcular la media macro excluyendo los NaN
    valid_scores = [score for score in auroc_scores.values() if not np.isnan(score)]
    macro_auroc = np.mean(valid_scores) if valid_scores else float('nan')
    
    return {
        'macro_auroc': macro_auroc,
        'class_auroc': auroc_scores
    }
