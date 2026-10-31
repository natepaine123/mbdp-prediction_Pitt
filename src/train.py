import yaml
import joblib
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def main():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load Data directly from the processed CSV
    log.info("Loading data from CSV...")
    csv_path = config["data"]["train_path"]
    df = pd.read_csv(csv_path)
    
    target_col = config["data"]["target_column"]
    
    # Get the features that actually exist in the CSV
    features = [col for col in config["feature_columns"] if col in df.columns]
    
    X = df[features]
    y = df[target_col]

    log.info(f"Loaded {len(X)} patients with {len(features)} features: {features}")
    log.info(f"MBDP Positive: {y.sum()} | MBDP Negative: {len(y) - y.sum()}")

    # 3. Train/Test Split
    log.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["data"].get("random_state", 42),
        stratify=y
    )

    # 4. Train Model
    log.info("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=config["model"].get("max_iter", 1000),
        class_weight=config["model"].get("class_weight", "balanced"),
        random_state=config["model"].get("random_state", 42)
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    log.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*40)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("="*40)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No MBDP", "MBDP"], zero_division=0))
    print("="*40 + "\n")

    # 6. Save Model Bundle
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "mbdp_model.joblib"
    threshold_path = out_dir / "threshold.joblib"
    
    bundle = {
        "model": model,
        "features": features,
        "target": target_col
    }
    
    joblib.dump(bundle, model_path)
    joblib.dump(0.5, threshold_path)
    log.info(f"Model bundle saved to {model_path}")

if __name__ == "__main__":
    main()