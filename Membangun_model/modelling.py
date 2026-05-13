import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Breast Cancer Classification")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'breast_cancer_preprocessed.csv')

def train_model():
    print("Loading preprocessed data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="baseline_random_forest"):
        print("Training Random Forest...")
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        print(f"Accuracy: {score:.4f}")
        print("Model berhasil dilatih dan artefak tersimpan di MLflow Tracking UI.")

if __name__ == "__main__":
    train_model()
