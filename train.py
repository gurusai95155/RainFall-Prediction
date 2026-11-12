import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def train_and_register():
    print("Loading data from Rainfall.csv...")
    df = pd.read_csv("Rainfall.csv")
    
    # Preprocessing
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['day'])
    df['winddirection'] = df['winddirection'].fillna(df['winddirection'].mode()[0])
    df['windspeed'] = df['windspeed'].fillna(df['windspeed'].median())
    df['rainfall'] = df['rainfall'].map({"yes": 1, "no": 0})
    df = df.drop(columns=['maxtemp', 'temparature', 'mintemp'])
    
    # Downsampling majority class to balance
    np.random.seed(42)
    df_majority = df[df['rainfall'] == 1]
    df_minority = df[df['rainfall'] == 0]
    downsample_indices = np.random.choice(df_majority.index, size=len(df_minority), replace=False)
    df_majority_downsampled = df_majority.loc[downsample_indices]
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_downsampled.drop(columns=['rainfall'])
    y = df_downsampled['rainfall']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Model trained successfully! Test Accuracy: {acc:.4f}")
    
    # Save standalone pickle for resilience
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Saved local standalone model to {model_path}")
    
    # MLflow tracking and registration
    # Setup local SQLite tracking URI for MLflow Model Registry
    backend_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(backend_uri)
    mlflow.set_experiment("Rainfall1")
    
    # Register model in MLflow Model Registry
    client = MlflowClient(tracking_uri=backend_uri)
    model_name = "rainfall-prediction-production"
    
    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": 50,
            "max_depth": "None",
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "min_samples_split": 10
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "recall_class_0": report['0']['recall'],
            "recall_class_1": report['1']['recall'],
            "f1_score_macro": report['macro avg']['f1-score']
        })
        
        # Log model and automatically register with MLflow registry
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="champion_model",
            registered_model_name=model_name
        )
        run_id = run.info.run_id
        print(f"MLflow Run logged with ID: {run_id}")
    
    # Set alias @champion on the latest version
    latest_versions = client.get_registered_model(model_name).latest_versions
    if latest_versions:
        latest_version = latest_versions[-1].version
        client.set_registered_model_alias(name=model_name, alias="champion", version=latest_version)
        print(f"Set alias '@champion' on model version {latest_version} for {model_name}")

if __name__ == '__main__':
    train_and_register()
