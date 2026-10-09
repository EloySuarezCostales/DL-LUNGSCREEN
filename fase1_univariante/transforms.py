import torchvision.transforms as transforms

# Valores promedio de ImageNet colapsados a 1 canal (escala de grises médica)
# Original RGB - Mean: [0.485, 0.456, 0.406] -> Avg: 0.449
# Original RGB - Std: [0.229, 0.224, 0.225] -> Avg: 0.226
GRAY_MEAN = [(0.485 + 0.456 + 0.406) / 3]
GRAY_STD = [(0.229 + 0.224 + 0.225) / 3]

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Pipeline de transformaciones para el conjunto de ENTRENAMIENTO.
    Incluye Data Augmentation conservador adaptado a radiografías médicas.
    IMPORTANTE: No se incluye RandomHorizontalFlip porque la asimetría del 
    corazón (Cardiomegalia) y la posición gástrica son críticas en el diagnóstico.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Pequeñas rotaciones y traslaciones (simula diferencias de posicionamiento del paciente)
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # Ajustes sutiles de contraste y brillo (simula diferencias de exposición del tubo de rayos X)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
    ])

def get_valid_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Pipeline de transformaciones para el conjunto de VALIDACIÓN.
    Estrictamente determinista (sin ruido).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
    ])
