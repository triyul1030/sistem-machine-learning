import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

try:
    print("Fetching breast cancer dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    output_path = r"c:\Users\ACER\Documents\Projects\Sistem-Machine-Learning\dataset_raw\breast_cancer.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset berhasil disimpan ke {output_path}")
except Exception as e:
    print(f"Error: {e}")
