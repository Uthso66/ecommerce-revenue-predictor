import pandas as pd
import joblib
import yaml
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(processed_dir):
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv')).squeeze()
    return X_train, y_train, X_val, y_val

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)
    print(f"\n✅ Validation Evaluation:\nMAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
    return mae, mse, rmse, r2

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n💾 Model saved to: {path}")

def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    model_path = config['model']['save_path']

    X_train, y_train, X_val, y_val = load_data(processed_dir)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    save_model(model, model_path)

if __name__ == "__main__":
    main()
