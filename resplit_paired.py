import pandas as pd
from sklearn.model_selection import train_test_split
import os

def extract_study_id(path):
    parts = str(path).split('/')
    if len(parts) >= 4:
        return "/".join(parts[1:4])
    return str(path)

def main():
    print("[*] Cargando CSVs originales...")
    df_train = pd.read_csv('train_split_85.csv')
    df_valid = pd.read_csv('valid_split_15.csv')
    
    # Combinar todo
    df_all = pd.concat([df_train, df_valid], ignore_index=True)
    
    # Extraer Study_ID
    df_all['Study_ID'] = df_all['Path'].apply(extract_study_id)
    
    label_cols = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    print("[*] Emparejando estudios...")
    frontals = df_all[df_all['Frontal/Lateral'] == 'Frontal'].copy()
    laterals = df_all[df_all['Frontal/Lateral'] == 'Lateral'].copy()
    
    frontals = frontals.drop_duplicates(subset=['Study_ID'], keep='first')
    laterals = laterals.drop_duplicates(subset=['Study_ID'], keep='first')
    
    frontals = frontals.rename(columns={'Path': 'Path_Frontal'})
    laterals = laterals.rename(columns={'Path': 'Path_Lateral'})
    
    # Merge para quedarnos solo con estudios completos
    merged = pd.merge(
        frontals[['Study_ID', 'Path_Frontal'] + label_cols],
        laterals[['Study_ID', 'Path_Lateral']],
        on='Study_ID',
        how='inner'
    )
    
    print(f"[*] Total de estudios pareados encontrados: {len(merged)}")
    
    # Vamos a estratificar basándonos en "Lung Lesion" y "Lung Opacity", que son las clases críticas.
    # Convertimos los nan a 0 y -1 a 1 temporalmente solo para la estratificación
    strat_labels = merged['Lung Lesion'].fillna(0).replace(-1.0, 1.0).astype(int).astype(str) + "_" + merged['Lung Opacity'].fillna(0).replace(-1.0, 1.0).astype(int).astype(str)
    
    print("[*] Realizando split 85/15 estratificado...")
    train_df, valid_df = train_test_split(
        merged, 
        test_size=0.15, 
        random_state=42, 
        stratify=strat_labels
    )
    
    train_df.to_csv('train_paired.csv', index=False)
    valid_df.to_csv('valid_paired.csv', index=False)
    
    print(f"[+] Guardado train_paired.csv con {len(train_df)} estudios.")
    print(f"[+] Guardado valid_paired.csv con {len(valid_df)} estudios.")
    
    # Mostrar conteos de validación para verificar
    print("\n[Distribución en Validación]")
    for col in ['Lung Lesion', 'Lung Opacity', 'Cardiomegaly', 'Consolidation', 'Pneumonia']:
        pos = (valid_df[col] == 1.0).sum()
        print(f"  {col}: {pos} casos positivos")

if __name__ == "__main__":
    main()
