import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_path = 'breast_cancer_preprocessed.csv'

def train_model():
    print("Loading preprocessed data for MLProject CI...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please ensure data is copied here.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"Accuracy: {score:.4f}")

if __name__ == "__main__":
    train_model()
