import pandas as pd
from sklearn.model_selection import train_test_split
import logging
log = logging.getLogger(__name__)

def load_data(config: dict):
    df = pd.read_csv(config["data"]["train_path"])
    target = config["data"]["target_column"]
    if target not in df.columns:
        raise KeyError(f"Target '{target}' missing from {config['data']['train_path']}. "
                       f"Run src.real_data_loader first.")

    present = [c for c in config["feature_columns"] if c in df.columns]
    missing = [c for c in config["feature_columns"] if c not in df.columns]
    if missing:
        log.warning("Dropping configured features absent from data: %s", missing)
    if not present:
        raise ValueError("No configured features exist in the CSV.")

    X, y = df[present], df[target]
    return train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    ), present