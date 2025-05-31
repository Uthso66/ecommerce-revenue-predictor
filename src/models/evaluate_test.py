import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.plot_residuals import plot_residuals, plot_prediction_distribution

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n🧪 Test Evaluations:")
    print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R²:{r2:.2f}")

    return predictions

def main():
    print("📦 Loading processed test data...")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    print("📦 Loading trained model...")
    model = joblib.load("models/housing_model.pkl")
    print("📊 Evaluating on test set...")
    test_preds = evaluate_model(model, X_test, y_test)
    
    plot_residuals(y_test, test_preds, output_path="outputs/residuals_plot.png")
    plot_prediction_distribution(y_test, test_preds, output_path="outputs/prediction_dist.png")    

if __name__ == "__main__":
    main()