import yaml
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.preprocessing import load_and_preprocess_data
from src.utils import find_threshold_for_sensitivity

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_preprocess_data(config)

    model = LogisticRegression(
        class_weight=config['model']['class_weight'],
        solver=config['model']['solver'],
        max_iter=config['model']['max_iter'],
        random_state=config['model']['random_state']
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Validation AUC: {auc:.4f}")

    target_sens = config['threshold']['target_sensitivity']
    threshold = find_threshold_for_sensitivity(y_val, y_val_pred_proba, target_sens)
    print(f"Threshold for {target_sens*100:.1f}% sensitivity: {threshold:.4f}")

    y_val_pred = (y_val_pred_proba >= threshold).astype(int)
    print("Validation classification report:")
    print(classification_report(y_val, y_val_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    model_path = os.path.join(config['paths']['model_dir'], config['paths']['model_name'])
    threshold_path = os.path.join(config['paths']['model_dir'], config['paths']['threshold_name'])
    joblib.dump(pipeline, model_path)
    joblib.dump(threshold, threshold_path)
    print(f"Model saved to {model_path}")
    print(f"Threshold saved to {threshold_path}")

    y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)
    print("\nTest set performance:")
    print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    main()