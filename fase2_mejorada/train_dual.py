import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

# Importaciones locales de los módulos de la Fase 2
from dataset_paired import PairedCheXpertDataset, CHEXPERT_TASKS
from transforms_dual import get_transforms_frontal, get_transforms_lateral
from models_fusion import DualStreamDenseNet
from utils import compute_auroc

# Hiperparámetros defensivos OOM
BATCH_SIZE = 16 
GRADIENT_ACCUMULATION_STEPS = 2  
NUM_EPOCHS = 15

def calculate_pos_weights(dataset: PairedCheXpertDataset) -> torch.Tensor:
    print("[*] Calculando pesos de balanceo para BCEWithLogitsLoss...")
    labels = dataset.labels
    pos_weights = []
    
    for i in range(labels.shape[1]):
        positives = (labels[:, i] == 1.0).sum().item()
        negatives = (labels[:, i] == 0.0).sum().item()
        weight = negatives / positives if positives > 0 else 1.0
        pos_weights.append(weight)
        
    return torch.FloatTensor(pos_weights)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (img_front, img_lat, labels) in enumerate(dataloader):
        img_front, img_lat, labels = img_front.to(device), img_lat.to(device), labels.to(device)
        
        # Máscara para política U-Ignore
        valid_mask = (labels != -1.0).float()
        safe_labels = labels.clone()
        safe_labels[safe_labels == -1.0] = 0.0

        logits = model(img_front, img_lat)
        
        # Pérdida enmascarada individualmente
        loss_matrix = criterion(logits, safe_labels)
        masked_loss = loss_matrix * valid_mask
        loss = (masked_loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)) / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
        running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        if (i + 1) % 100 == 0:
            print(f"   Batch {i+1}/{len(dataloader)} - Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for img_front, img_lat, labels in dataloader:
            img_front, img_lat, labels = img_front.to(device), img_lat.to(device), labels.to(device)
            
            valid_mask = (labels != -1.0).float()
            safe_labels = labels.clone()
            safe_labels[safe_labels == -1.0] = 0.0
            
            logits = model(img_front, img_lat)
            
            loss_matrix = criterion(logits, safe_labels)
            masked_loss = loss_matrix * valid_mask
            loss = masked_loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # compute_auroc ya procesa la política U-Ignore extrayendo dinámicamente los -1.0
    auroc_results = compute_auroc(all_labels, all_preds, CHEXPERT_TASKS)
    macro_auroc = auroc_results['macro_auroc']
    
    return running_loss / len(dataloader), macro_auroc, auroc_results

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento Dual-Stream SOTA (Cross-Attention + Unfreezing)")
    parser.add_argument('--resume', type=str, default=None, help="Ruta al checkpoint (.pth) para reanudar el entrenamiento")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Número de épocas a entrenar en total")
    parser.add_argument('--warmup_epochs', type=int, default=3, help="Épocas con backbones congelados")
    parser.add_argument('--lr_warmup', type=float, default=1e-4, help="Learning Rate para el Cerebro")
    parser.add_argument('--lr_finetune', type=float, default=1e-5, help="Learning Rate para toda la red tras descongelar")
    parser.add_argument('--zip_path', type=str, default='/kaggle/input/datasets/ashery/chexpert', help="Ruta al ZIP")
    parser.add_argument('--train_csv', type=str, default='train_paired.csv', help="Ruta al CSV de entrenamiento")
    parser.add_argument('--valid_csv', type=str, default='valid_paired.csv', help="Ruta al CSV de validación")
    parser.add_argument('--frontal_ckpt', type=str, default='densenet_frontal_best.pth', help="Ruta al checkpoint frontal")
    parser.add_argument('--lateral_ckpt', type=str, default='densenet_lateral_best.pth', help="Ruta al checkpoint lateral")
    parser.add_argument('--uncertainty', type=str, default='U-Ignore', choices=['U-Ones', 'U-Zeroes', 'U-Ignore'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dispositivo de entrenamiento: {device}")
    
    print("[*] Configurando DataLoaders...")
    train_dataset = PairedCheXpertDataset(
        csv_path=args.train_csv, zip_path=args.zip_path, uncertainty_policy=args.uncertainty,
        transform_frontal=get_transforms_frontal(is_train=True), transform_lateral=get_transforms_lateral(is_train=True)
    )
    valid_dataset = PairedCheXpertDataset(
        csv_path=args.valid_csv, zip_path=args.zip_path, uncertainty_policy=args.uncertainty,
        transform_frontal=get_transforms_frontal(is_train=False), transform_lateral=get_transforms_lateral(is_train=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    pos_weights = calculate_pos_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    
    print("[*] Instanciando Arquitectura SOTA Bimodal (Cross-Attention)...")
    frontal_path = args.frontal_ckpt if os.path.exists(args.frontal_ckpt) else None
    lateral_path = args.lateral_ckpt if os.path.exists(args.lateral_ckpt) else None
    
    model = DualStreamDenseNet(
        num_classes=14, frontal_checkpoint_path=frontal_path, lateral_checkpoint_path=lateral_path, freeze_backbones=True
    ).to(device)
    
    # 1. Optimizador Inicial (Solo Cerebro)
    optimizer = torch.optim.Adam(model.fusion_head.parameters(), lr=args.lr_warmup)
    
    # Scheduler Inicial
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    start_epoch = 0
    best_macro_auroc = 0.0
    backbones_unfrozen = False
    
    # --- LÓGICA DE REANUDACIÓN (RESUME) ---
    if args.resume and os.path.exists(args.resume):
        print(f"[*] Reanudando entrenamiento desde: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_macro_auroc = checkpoint['best_macro_auroc']
        backbones_unfrozen = checkpoint.get('backbones_unfrozen', False)
        
        if backbones_unfrozen or start_epoch >= args.warmup_epochs:
            model.unfreeze_backbones()
            backbones_unfrozen = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[+] Reanudado en la época {start_epoch+1} con mejor AUROC previo de {best_macro_auroc:.4f}")

    print(f"[*] Iniciando entrenamiento ({args.epochs} épocas solicitadas)...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Época {epoch+1}/{args.epochs} ---")
        
        # --- DESCONGELACIÓN GRADUAL (WARM-UP a FINE-TUNING) ---
        if epoch == args.warmup_epochs and not backbones_unfrozen:
            print("[!] Iniciando Fase 2 (Fine-Tuning End-to-End). Descongelando backbones...")
            model.unfreeze_backbones()
            backbones_unfrozen = True
            
            # Recreamos el optimizador para incluir TODA la red neuronal, a un ritmo microscópico
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            print(f"[!] Nuevo Learning Rate ajustado a {args.lr_finetune} para proteger el conocimiento previo.")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, macro_auroc, auroc_results = validate(model, valid_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Macro AUROC: {macro_auroc:.4f}")
        
        # Detalle de las clases más críticas
        class_auroc = auroc_results.get('class_auroc', {})
        print(f"Detalle: Lung Lesion AUROC: {class_auroc.get('Lung Lesion', 'N/A'):.4f} | "
              f"Lung Opacity AUROC: {class_auroc.get('Lung Opacity', 'N/A'):.4f}")
        
        # Le damos el Macro AUROC al Scheduler para que decida si frenar el LR
        scheduler.step(macro_auroc)
        
        # 1. Guardar el MEJOR modelo (limpio)
        if macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            torch.save(model.state_dict(), 'dual_stream_best.pth')
            print("[+] Nuevo modelo SOTA guardado!")
            
        # 2. Guardar el estado ACTUAL (para reanudar)
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_macro_auroc': best_macro_auroc,
            'backbones_unfrozen': backbones_unfrozen
        }
        torch.save(checkpoint_state, 'dual_stream_resume.pth')
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
