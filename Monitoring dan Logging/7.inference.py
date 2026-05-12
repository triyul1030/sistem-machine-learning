import requests
import json
import pandas as pd
import random
import time

url = 'http://127.0.0.1:8080/invocations'

kolom_breast_cancer = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def send_request():
    dummy_data = [[random.uniform(-2, 2) for _ in range(30)]]
    
    headers = {
        'Content-Type': 'application/json',
    }

    payload = {
        "dataframe_split": {
            "columns": kolom_breast_cancer,
            "data": dummy_data
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"Success: {response.json()}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Gagal terhubung.")

if __name__ == "__main__":
    print("Mengirimkan simulasi inference requests...")
    for i in range(5):
        send_request()
        time.sleep(1)
