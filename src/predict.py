"""
Predict script — EXACT assignment format.
"""
import argparse, os, pickle, numpy as np, pandas as pd
from data_preprocessing import Preprocessor  # ensures class is importable

# match the expansion used in training
def poly_expand(X, degree=2):
    bias, F = X[:,[0]], X[:,1:]
    parts = [F] + [F**d for d in range(2, degree+1)]
    return np.hstack([bias] + parts)

def predict_from_dict(model, X):
    t = model["type"]
    if t in ("ols", "ridge", "lasso", "elasticnet"):
        return X @ model["w"]
    if t == "poly_ols" or t == "poly_ridge":
        return poly_expand(X, model["degree"]) @ model["w"]
    raise ValueError(f"Unknown model type: {t}")

def ensure_dir(p):
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mse, rmse, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--metrics_output_path", required=True)
    ap.add_argument("--predictions_output_path", required=True)
    args = ap.parse_args()

    with open(args.model_path, "rb") as f:
        bundle = pickle.load(f)  # {'model': dict, 'preprocessor': Preprocessor}
    model = bundle["model"]; pre = bundle["preprocessor"]

    df = pd.read_csv(args.data_path)
    X, y_true = pre.transform(df)
    y_pred = predict_from_dict(model, X)

    ensure_dir(args.predictions_output_path)
    pd.DataFrame(y_pred).to_csv(args.predictions_output_path, index=False, header=False)

    mse, rmse, r2 = metrics(y_true, y_pred)
    ensure_dir(args.metrics_output_path)
    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")

    print("Predictions and metrics written to /results")

if __name__ == "__main__":
    main()

