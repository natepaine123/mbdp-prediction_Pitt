import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(config):
    """
    Load data, split into train/validation/test, and create preprocessing pipeline.
    """
    try:
        df = pd.read_csv(config['data']['train_path'])
    except FileNotFoundError:
        print("Data file not found. Generating synthetic data...")
        from src.data_generator import generate_synthetic_data
        df = generate_synthetic_data(10000)
        df.to_csv(config['data']['train_path'], index=False)

    target = config['data']['target_column']
    feature_cols = config['data']['feature_columns']
    categorical_cols = config['data']['categorical_columns']

    X = df[feature_cols]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=config['model']['random_state'], stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config['model']['random_state'], stratify=y_temp
    )

    numeric_features = [c for c in feature_cols if c not in categorical_cols]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor