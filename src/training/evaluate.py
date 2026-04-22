import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from mlflow.tracking import MlflowClient
from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger
from src.training.utils import build_drop_columns

logger = get_logger(__name__)

# Load central configs
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]

def _load_and_prep_val_data():
    """Helper to load validation data consistently for Churn."""
    val_path = f"{get_path('splits')}/val.parquet"
    drop_columns = build_drop_columns(TRAIN_CFG)
    # clean_names makes columns lowercase
    target_col = TRAIN_CFG["data"]["target_column"].lower().replace(" ", "_")
    
    val_df = pd.read_parquet(val_path)
    X_val = val_df.drop(columns=drop_columns, errors="ignore")
    # Numeric mapping for metrics
    y_val = (
        val_df[target_col]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )
    return X_val, y_val

def _generate_and_log_plots(model, X_val, y_val, run_id):
    """Generates evaluation plots and logs them to the specific MLflow run."""
    logger.info(f"Generating evaluation plots for run {run_id}...")
    
    # 1. Predict (Handle pyfunc potential probabilistic output)
    preds = model.predict(X_val)
    preds = (preds > 0.5).astype(int) if preds.dtype == float else preds
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close() # Important to free memory
    
    # 3. Feature Importance Plot (if model supports it)
    # Note: For pyfunc, access to feature_importances_ can be complex.
    # A common workaround is to log feature importance during train.py
    # or to unwrap the model. For simplicity in this blueprint, 
    # we focus on the confusion matrix which is always available.
    # To add feature importance, you'd need framework-specific unwrapping.

    # 4. Log Artifacts to the specific run
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_artifact(cm_path, "evaluation/plots")
        logger.info(f"Confusion matrix plot logged to run {run_id}.")
        
    # Clean up local file
    if os.path.exists(cm_path):
        os.remove(cm_path)

def evaluate_model(model_alias: str = "champion") -> float:
    """
    Evaluates a specific model from the registry on the current validation set.
    Returns the F1-Score (instead of RMSE).
    """
    X_val, y_val = _load_and_prep_val_data()

    try:
        model_uri = f"models:/{MODEL_NAME}@{model_alias}"
        logger.info(f"Evaluating {model_alias} from registry: {model_uri}")
        
        # Using pyfunc to be framework-agnostic (works for XGB, Sklearn, etc.)
        model = mlflow.pyfunc.load_model(model_uri)
        preds = model.predict(X_val)
        
        # Handle potential probability outputs from pyfunc
        preds = (preds > 0.5).astype(int) if preds.dtype == float else preds
        
        f1 = f1_score(y_val, preds)
        logger.info(f"Model {model_alias} F1-Score: {f1:.4f}")
        return float(f1)
        
    except Exception as e:
        logger.warning(f"Could not evaluate {model_alias}: {e}")
        return None

def compare_models(new_run_id: str, val_path: str | None = None):
    """
    Compares the new model (Challenger) with the current Champion.
    Also generates detailed plots for the Challenger.
    Returns (is_better: bool, metrics: dict).
    """
    client = MlflowClient()
    X_val, y_val = _load_and_prep_val_data()
    
    # 1. Evaluate and Plot the Challenger
    logger.info(f"Evaluating Challenger (Run ID: {new_run_id})...")
    challenger_uri = f"runs:/{new_run_id}/model"
    challenger = mlflow.pyfunc.load_model(challenger_uri)
    
    # Predict for metric calculation
    chall_preds = challenger.predict(X_val)
    chall_preds = (chall_preds > 0.5).astype(int) if chall_preds.dtype == float else chall_preds
    
    chall_f1 = f1_score(y_val, chall_preds)
    metrics = {"challenger_f1": float(chall_f1)}
    
    # --- ADDED: Generate and log plots ---
    _generate_and_log_plots(challenger, X_val, y_val, new_run_id)

    # 2. Evaluate the current Champion
    try:
        champion_uri = f"models:/{MODEL_NAME}@champion"
        logger.info(f"Evaluating current Champion from Registry: {champion_uri}")
        
        champion = mlflow.pyfunc.load_model(champion_uri)
        champ_preds = champion.predict(X_val)
        champ_preds = (champ_preds > 0.5).astype(int) if champ_preds.dtype == float else champ_preds
        
        champ_f1 = f1_score(y_val, champ_preds)
        metrics["champion_f1"] = float(champ_f1)
        
        logger.info(f"Comparison: Challenger F1 ({chall_f1:.4f}) vs Champion F1 ({champ_f1:.4f})")

        # For Churn/F1: Higher is better!
        is_better = chall_f1 > champ_f1
        return is_better, metrics

    except Exception as e:
        logger.warning(f"Comparison skipped (Reason: {e}). Challenger wins by default.")
        return True, metrics

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)