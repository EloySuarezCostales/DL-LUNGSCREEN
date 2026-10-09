import torchvision.transforms as transforms

# Normalización para 1 canal (escala de grises) adaptada de ImageNet, idéntico a Fase 1
GRAY_MEAN = [(0.485 + 0.456 + 0.406) / 3]
GRAY_STD = [(0.229 + 0.224 + 0.225) / 3]
IMAGE_SIZE = 320

def get_transforms_frontal(is_train: bool) -> transforms.Compose:
    """
    Pipeline de transformaciones para la imagen FRONTAL.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
        ])

def get_transforms_lateral(is_train: bool) -> transforms.Compose:
    """
    Pipeline de transformaciones para la imagen LATERAL.
    INDENDIENTE para forzar asincronía espacial.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(-7, 7)),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)
        ])
