import pandas as pd
import numpy as np
import mlflow
import dagshub
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dagshub.init(repo_owner='triyul1030', repo_name='sistem-machine-learning', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/triyul1030/sistem-machine-learning.mlflow")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
data_path = os.path.join(root_dir, 'preprocessing', 'dataset_preprocessing', 'breast_cancer_preprocessed.csv')

def train_model():
    print("Loading preprocessed data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.sklearn.autolog(disable=True)

    mlflow.set_experiment("Model_Tuning")
    
    with mlflow.start_run(run_name="tuned_random_forest"):
        print("Starting Hyperparameter Tuning with GridSearchCV...")
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        mlflow.sklearn.log_model(best_model, "model")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = os.path.join(script_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(8,6))
        plt.title('Top 10 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')

        plt.tight_layout()

        fi_path = os.path.join(script_dir, 'feature_importance.png')

        plt.savefig(fi_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path)
        
        print("Tuning dan manual logging selesai!")

if __name__ == "__main__":
    train_model()
