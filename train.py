import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Importar los módulos que hemos construido
from dataset import CheXpertDataset, CHEXPERT_TASKS
from models import CheXpertDenseNet
from utils import calculate_pos_weights, compute_auroc
from transforms import get_train_transforms, get_valid_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Fase 1: Entrenamiento Univariante CheXpert")
    parser.add_argument('--view', type=str, required=True, choices=['Frontal', 'Lateral'],
                        help="Vista a entrenar: 'Frontal' o 'Lateral'")
    parser.add_argument('--zip_path', type=str, default='archive.zip', help="Ruta al ZIP del dataset")
    parser.add_argument('--train_csv', type=str, default='train.csv', help="Ruta a los metadatos de entrenamiento")
    parser.add_argument('--valid_csv', type=str, default='valid.csv', help="Ruta a los metadatos de validación")
    parser.add_argument('--batch_size', type=int, default=16, help="Tamaño del lote (reducir si falta VRAM)")
    parser.add_argument('--epochs', type=int, default=5, help="Número de épocas de entrenamiento")
    parser.add_argument('--lr', type=float, default=1e-4, help="Tasa de aprendizaje (Learning Rate)")
    parser.add_argument('--uncertainty', type=str, default='U-Ignore', choices=['U-Ones', 'U-Zeroes', 'U-Ignore'],
                        help="Política para las etiquetas inciertas -1.0")
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Bucle de entrenamiento de una época.
    """
    model.train()
    running_loss = 0.0
    
    # tqdm genera una barra de progreso visual en consola
    for images, labels in tqdm(dataloader, desc="Entrenamiento"):
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Máscara para ignorar los diagnósticos dudosos (-1.0)
        valid_mask = (labels != -1.0).float()
        
        # 2. Reemplazo temporal a 0.0 para que BCEWithLogitsLoss no dé error interno.
        # Sus errores serán anulados por la máscara justo después.
        safe_labels = labels.clone()
        safe_labels[safe_labels == -1.0] = 0.0

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # 3. Pérdida individual (matriz completa)
        loss_matrix = criterion(outputs, safe_labels)
        
        # 4. Enmascaramos la pérdida (los -1.0 pasan a tener coste 0)
        masked_loss = loss_matrix * valid_mask
        
        # 5. Calculamos la media solo entre los elementos válidos
        loss = masked_loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)
        
        # Backward pass y optimización
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """
    Bucle de validación de una época. 
    Usa torch.no_grad() para no acumular gradientes y ahorrar memoria VRAM.
    """
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Validación"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Máscara para la validación
        valid_mask = (labels != -1.0).float()
        safe_labels = labels.clone()
        safe_labels[safe_labels == -1.0] = 0.0
        
        outputs = model(images)
        loss_matrix = criterion(outputs, safe_labels)
        masked_loss = loss_matrix * valid_mask
        loss = masked_loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)
        
        running_loss += loss.item()
        
        # Aplicamos la función sigmoide para convertir logits en probabilidades (0 a 1)
        # Esto es obligatorio antes de calcular el AUROC
        probs = torch.sigmoid(outputs)
        
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    avg_loss = running_loss / len(dataloader)
    
    # Concatenar todos los lotes para calcular métricas globales
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    auroc_results = compute_auroc(all_labels, all_preds, CHEXPERT_TASKS)
    
    return avg_loss, auroc_results

def main():
    args = parse_args()
    
    # Detección automática de Tarjeta Gráfica (GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[*] Dispositivo de cálculo: {device}")
    print(f"[*] Inicializando entorno para el modelo: {args.view.upper()}")
    
    # 1. Instanciar Datasets
    print("[*] Conectando con los datos en formato ZIP (Lazy Loading)...")
    train_dataset = CheXpertDataset(
        csv_path=args.train_csv,
        zip_path=args.zip_path,
        view_type=args.view,
        uncertainty_policy=args.uncertainty,
        transform=get_train_transforms(image_size=224)
    )
    
    valid_dataset = CheXpertDataset(
        csv_path=args.valid_csv,
        zip_path=args.zip_path,
        view_type=args.view,
        uncertainty_policy=args.uncertainty,
        transform=get_valid_transforms(image_size=224)
    )
    
    # 2. Instanciar DataLoaders (Optimizados para velocidad)
    # En Windows num_workers > 0 puede ser inestable con ciertos paquetes,
    # lo dejamos en 0 por defecto para garantizar estabilidad inmediata.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 3. Calcular Balance de Clases
    print(f"[*] Calculando pesos para balancear la pérdida (pos_weights) en {len(CHEXPERT_TASKS)} clases...")
    pos_weights = calculate_pos_weights(
        csv_path=args.train_csv,
        target_columns=CHEXPERT_TASKS,
        view_type=args.view,
        uncertainty_policy=args.uncertainty
    ).to(device)
    
    # 4. Preparar Red Neuronal y Motor de Entrenamiento
    print("[*] Desplegando arquitectura DenseNet-121 de 1 canal...")
    model = CheXpertDenseNet(num_classes=len(CHEXPERT_TASKS)).to(device)
    
    # Se introduce el peso para ayudar a la red con las enfermedades raras
    # Usamos reduction='none' para permitir el enmascaramiento dinámico (U-Ignore)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_macro_auroc = 0.0
    
    # 5. Bucle Principal (Épocas)
    print("\n[+] === INICIANDO ENTRENAMIENTO ===")
    for epoch in range(1, args.epochs + 1):
        print(f"\nÉpoca {epoch}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, auroc_results = validate_one_epoch(model, valid_loader, criterion, device)
        
        macro_auroc = auroc_results['macro_auroc']
        
        print(f" -> Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Macro AUROC: {macro_auroc:.4f}")
        
        # Imprimir desglose de las clases de mayor interés para ti
        print(f" -> Detalle: Lung Lesion AUROC: {auroc_results['class_auroc'].get('Lung Lesion', 'N/A'):.4f} | "
              f"Lung Opacity AUROC: {auroc_results['class_auroc'].get('Lung Opacity', 'N/A'):.4f}")
        
        # Guardar el modelo si bate el récord (Early Stopping básico)
        if macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            save_path = f"checkpoints/densenet_{args.view.lower()}_best.pth"
            
            # Guardamos un diccionario (state_dict) en lugar del modelo entero
            # Esta es la práctica recomendada en PyTorch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_auroc': macro_auroc,
                'view_type': args.view,
                'classes': CHEXPERT_TASKS
            }, save_path)
            
            print(f" [!] *Nuevo modelo estrella guardado* ({save_path})")

    print("\n[+] ENTRENAMIENTO FINALIZADO CON ÉXITO.")

if __name__ == "__main__":
    main()
