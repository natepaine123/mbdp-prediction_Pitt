import yaml
import joblib
import pandas as pd
import argparse
import os

def load_model_and_threshold(config):
    model_path = os.path.join(config['paths']['model_dir'], config['paths']['model_name'])
    threshold_path = os.path.join(config['paths']['model_dir'], config['paths']['threshold_name'])
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    return model, threshold

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

    model, threshold = load_model_and_threshold(config)

    input_df = pd.read_csv(args.input)
    feature_cols = config['data']['feature_columns']
    X = input_df[feature_cols]

    result = predict(X, model, threshold)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Flagged {result['mbdp_flag'].sum()} out of {len(result)} patients.")

if __name__ == "__main__":
    main()