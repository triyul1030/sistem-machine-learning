import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss
)
from sklearn.utils import estimator_html_repr

dagshub.init(repo_owner='triyul1030', repo_name='sistem-machine-learning', mlflow=True)
mlflow.set_experiment("Breast Cancer Classification")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'breast_cancer_preprocessed.csv')

def train_model():
    print("Loading preprocessed data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name="tuned_random_forest"):
        print("Starting Hyperparameter Tuning with GridSearchCV...")
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        logloss = log_loss(y_test, y_pred_proba)

        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}, Log Loss: {logloss:.4f}")

        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "log_loss": logloss
        })

        mlflow.sklearn.log_model(best_model, "model")

        estimator_html = estimator_html_repr(best_model)
        estimator_path = os.path.join(script_dir, "estimator.html")
        with open(estimator_path, "w", encoding="utf-8") as f:
            f.write(estimator_html)
        mlflow.log_artifact(estimator_path)

        metric_info = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "log_loss": logloss
        }
        metric_info_path = os.path.join(script_dir, "metric_info.json")
        with open(metric_info_path, "w") as f:
            json.dump(metric_info, f, indent=2)
        mlflow.log_artifact(metric_info_path)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'])
        plt.title('Training Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = os.path.join(script_dir, "training_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)

        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(8, 6))
        plt.title('Top 10 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        fi_path = os.path.join(script_dir, "feature_importance.png")
        plt.savefig(fi_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path)

        report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
        report_path = os.path.join(script_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        print("Tuning dan manual logging selesai!")
        print("Artefak yang di-log: model/, estimator.html, metric_info.json, "
              "training_confusion_matrix.png, feature_importance.png, classification_report.txt")

if __name__ == "__main__":
    train_model()
