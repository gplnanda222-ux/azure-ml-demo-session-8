"""
Azure ML training script for loan approval model.
This script runs on Azure ML compute instances.
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train loan approval model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_type', type=str, default='random_forest', 
                       choices=['random_forest', 'logistic_regression'],
                       help='Model type to train')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for RF')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth for RF')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--use_feature_store', type=str, default='false', help='Use feature store (true/false)')
    
    return parser.parse_args()

def load_data(data_path):
    """Load preprocessed training data."""
    print(f"Looking for data in: {data_path}")
    print(f"Contents of data_path: {os.listdir(data_path) if os.path.exists(data_path) else 'Directory does not exist'}")
    
    # Try different possible data file locations
    possible_train_paths = [
        os.path.join(data_path, 'train_processed.csv'),
        os.path.join(data_path, 'processed_data', 'train_processed.csv'),
        os.path.join(data_path, 'train.csv'),
        os.path.join(data_path, 'data', 'train.csv'),
    ]
    
    possible_val_paths = [
        os.path.join(data_path, 'val_processed.csv'),
        os.path.join(data_path, 'processed_data', 'val_processed.csv'),
        os.path.join(data_path, 'val.csv'),
        os.path.join(data_path, 'data', 'val.csv'),
    ]
    
    # Find training data
    train_path = None
    for path in possible_train_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    if train_path is None:
        # List all CSV files in the directory tree
        csv_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        print(f"Available CSV files: {csv_files}")
        raise FileNotFoundError(f"Training data not found. Tried: {possible_train_paths}")
    
    print(f"Found training data: {train_path}")
    train_df = pd.read_csv(train_path)
    
    # Find validation data
    val_df = None
    for path in possible_val_paths:
        if os.path.exists(path):
            print(f"Found validation data: {path}")
            val_df = pd.read_csv(path)
            break
    
    print(f"Loaded training data: {len(train_df)} samples")
    if val_df is not None:
        print(f"Loaded validation data: {len(val_df)} samples")
    else:
        print("No validation data found, will split training data")
    
    return train_df, val_df

def prepare_features_target(df, target_col='loan_approved'):
    """Separate features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_model(X_train, y_train, model_type='random_forest', **hyperparams):
    """Train the selected model."""
    print(f"Training {model_type} model...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 10),
            random_state=hyperparams.get('random_state', 42),
            n_jobs=-1
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=hyperparams.get('random_state', 42),
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    print(f"Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test, dataset_name="test"):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    metrics = {
        f'{dataset_name}_accuracy': accuracy_score(y_test, y_pred),
        f'{dataset_name}_precision': precision_score(y_test, y_pred),
        f'{dataset_name}_recall': recall_score(y_test, y_pred),
        f'{dataset_name}_f1': f1_score(y_test, y_pred),
        f'{dataset_name}_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{dataset_name.title()} Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Classification report
    print(f"\n{dataset_name.title()} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_pred_proba

def plot_feature_importance(model, feature_names, output_dir):
    """Plot and save feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance")
        plt.barh(range(min(20, len(feature_names))), 
                importances[indices[:20]])
        plt.yticks(range(min(20, len(feature_names))), 
                  [feature_names[i] for i in indices[:20]])
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        print("Feature importance plot saved")

def plot_confusion_matrix(y_true, y_pred, output_dir, dataset_name="test"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name.title()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png'))
    plt.close()
    
    print(f"Confusion matrix plot saved for {dataset_name}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # MLflow is automatically handled by Azure ML
    # We don't need to manually configure it
    print("🔄 Running in Azure ML - MLflow tracking is automatic")
    use_mlflow = False  # Disable manual MLflow to avoid conflicts
    
    try:
        # Load data
        train_df, val_df = load_data(args.data_path)
        X_train, y_train = prepare_features_target(train_df)
        
        if val_df is not None:
            X_val, y_val = prepare_features_target(val_df)
        else:
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=args.random_state, stratify=y_train
            )
            print("Split training data: 80% train, 20% validation")
        
        # Log hyperparameters
        hyperparams = {
            'model_type': args.model_type,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'random_state': args.random_state
        }
        if use_mlflow:
            mlflow.log_params(hyperparams)
        
        # Train model
        model = train_model(X_train, y_train, **hyperparams)
        
        # Evaluate on training data
        train_metrics, _, _ = evaluate_model(model, X_train, y_train, "train")
        if use_mlflow:
            mlflow.log_metrics(train_metrics)
        
        # Evaluate on validation data (always available now)
        val_metrics, y_val_pred, y_val_proba = evaluate_model(model, X_val, y_val, "val")
        if use_mlflow:
            mlflow.log_metrics(val_metrics)
        
        # Plot confusion matrix for validation
        plot_confusion_matrix(y_val, y_val_pred, args.output_dir, "validation")
        
        # Plot feature importance
        feature_names = list(X_train.columns)
        plot_feature_importance(model, feature_names, args.output_dir)
        
        # Save model
        model_path = os.path.join(args.output_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Log model to MLflow
        if use_mlflow:
            mlflow.sklearn.log_model(model, "model")
        
        # Save model info
        model_info = {
            'model_type': args.model_type,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'hyperparameters': hyperparams,
            'metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        with open(os.path.join(args.output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Log artifacts
        if use_mlflow:
            mlflow.log_artifacts(args.output_dir)
        
        print("\nTraining completed successfully!")
        print(f"Model artifacts saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    
    finally:
        if use_mlflow:
            try:
                mlflow.end_run()
            except:
                pass

if __name__ == "__main__":
    main()