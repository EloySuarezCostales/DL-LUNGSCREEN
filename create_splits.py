import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import argparse

def main():
    parser = argparse.ArgumentParser(description="Crea particiones de entrenamiento y validación basadas en pacientes")
    parser.add_argument('--input_csv', type=str, default='train.csv', help="CSV original de entrenamiento")
    parser.add_argument('--train_out', type=str, default='train_split.csv', help="Nombre del nuevo CSV de entrenamiento")
    parser.add_argument('--valid_out', type=str, default='valid_split.csv', help="Nombre del nuevo CSV de validación")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proporción para validación (ej. 0.1 = 10%)")
    parser.add_argument('--seed', type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    args = parser.parse_args()

    print(f"[*] Cargando {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # 1. Extraer el ID del paciente de la ruta
    # Las rutas son del tipo: CheXpert-v1.0-small/train/patient00001/study1/...
    print("[*] Extrayendo identificadores de paciente...")
    df['Patient'] = df['Path'].str.extract(r'(patient\d+)')

    # Comprobación de seguridad
    if df['Patient'].isnull().any():
        print("[!] Advertencia: No se pudo extraer el paciente de todas las rutas.")
        # Eliminar las filas problemáticas si las hubiera para evitar problemas
        df = df.dropna(subset=['Patient'])

    # 2. Dividir agrupando por paciente (GroupShuffleSplit asegura que no haya "Patient leakage")
    print(f"[*] Dividiendo el dataset (Validación: {args.val_size*100}%)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    
    # next(gss.split(...)) devuelve los índices de train y validación
    train_idx, valid_idx = next(gss.split(df, groups=df['Patient']))

    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()

    # 3. Eliminar la columna temporal 'Patient' si lo deseamos (opcional)
    train_df = train_df.drop(columns=['Patient'])
    valid_df = valid_df.drop(columns=['Patient'])

    # 4. Guardar los nuevos CSV
    print(f"[*] Guardando {args.train_out} ({len(train_df)} imágenes)...")
    train_df.to_csv(args.train_out, index=False)
    
    print(f"[*] Guardando {args.valid_out} ({len(valid_df)} imágenes)...")
    valid_df.to_csv(args.valid_out, index=False)

    print("[+] Proceso completado con éxito.")
    
    # 5. Imprimir estadísticas rápidas sobre Lung Lesion
    print("\n[Estadísticas de Lung Lesion (1.0)]")
    train_ll = (train_df['Lung Lesion'] == 1.0).sum()
    valid_ll = (valid_df['Lung Lesion'] == 1.0).sum()
    print(f" -> Train: {train_ll} muestras positivas")
    print(f" -> Valid: {valid_ll} muestras positivas")

if __name__ == "__main__":
    main()
