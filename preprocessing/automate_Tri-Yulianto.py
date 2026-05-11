import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Mengecek missing values:")
    print(df.isnull().sum().sum())
    
    print("Menghapus data duplikat...")
    df = df.drop_duplicates()
    
    print("Memisahkan fitur dan target...")
    X = df.drop(columns=['target'])
    y = df['target']
    
    print("Standarisasi Fitur...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Menggabungkan kembali dengan target...")
    df_preprocessed = X_scaled_df.copy()
    df_preprocessed['target'] = y.values
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'breast_cancer_preprocessed.csv')
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Data preprocessed saved to: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(root_dir, 'dataset_raw', 'breast_cancer.csv')
    output_dir = os.path.join(script_dir, 'dataset_preprocessing')
    
    preprocess_data(input_path, output_dir)
