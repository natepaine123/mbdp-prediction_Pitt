import yaml
import joblib
import pandas as pd
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def load_model_and_threshold(config):
    model_path = "models/mbdp_model.joblib"
    threshold_path = "models/threshold.joblib"
    
    # 1. Load the bundle (dictionary)
    bundle = joblib.load(model_path)
    
    # 2. Extract the model and the exact features it was trained on
    model = bundle["model"]
    features = bundle["features"]
    
    # 3. Load threshold (with a fallback in case the file doesn't exist)
    if os.path.exists(threshold_path):
        threshold = joblib.load(threshold_path)
    else:
        log.warning("Threshold file not found. Defaulting to 0.5.")
        threshold = 0.5
        
    return model, features, threshold

def predict(input_df, model, threshold):
    proba = model.predict_proba(input_df)[:, 1]
    flag = (proba >= threshold).astype(int)
    result = input_df.copy()
    result['mbdp_risk_prob'] = proba
    result['mbdp_flag'] = flag
    return result

def main():
    parser = argparse.ArgumentParser(description="Predict MBDP risk for new data.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file with features.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output CSV file.")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Get model, features, and threshold
    model, features, threshold = load_model_and_threshold(config)

    input_df = pd.read_csv(args.input)
    
    # 4. Slice the input dataframe using the features saved by train.py
    # This prevents errors if the input CSV has extra columns or is missing unused ones
    try:
        X = input_df[features]
    except KeyError as e:
        log.error(f"Input data is missing required features: {e}")
        log.error(f"Model expects: {features}")
        return

    result = predict(X, model, threshold)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Flagged {result['mbdp_flag'].sum()} out of {len(result)} patients.")

if __name__ == "__main__":
    main()