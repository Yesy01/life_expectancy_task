#!/usr/bin/env python3
"""
data_pp1y.py - friend pipeline reproduced & aligned to your robust pipeline outputs.

Outputs saved (match robust pipeline expectations):
 - data/train_data_clean.csv
 - data/train_data_normalized.csv
 - data/processed/X_train.npy, X_val.npy, y_train.npy, y_val.npy
 - data/processed/X_train_scaled.npy, X_val_scaled.npy
 - models/preprocessor.pkl (dict with median, mad, feature_names, removed_features)
 - data/processed/metadata_pp1y.pkl

Usage:
    cd life_expectancy_task
    python3 src/data_pp1y.py
"""
import os
import numpy as np
import pandas as pd
import pickle

# -------------------------
# Helpers
# -------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def normalize_column_names(df):
    # strip whitespace, lowercase, replace spaces with underscore
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def fill_country_series(df, col):
    """Forward/backward fill within each country for a column."""
    return df.groupby("country")[col].transform(lambda g: g.ffill().bfill())

def mad(a, axis=0):
    med = np.nanmedian(a, axis=axis)
    return np.nanmedian(np.abs(a - med), axis=axis)

def robust_scale_train(X):
    med = np.nanmedian(X, axis=0)
    m = np.abs(X - med)
    mads = np.nanmedian(m, axis=0)
    # avoid zero mad
    mads = np.where(mads == 0, 1.0, mads)
    Xs = (X - med) / mads
    return Xs, med, mads

def robust_scale_apply(X, med, mads):
    Xs = (X - med) / mads
    return Xs

def cap_outliers_zscore(df_num, z_thresh=5.0):
    """
    Robustly cap outliers using median & MAD:
    - Compute z = (x - median) / MAD
    - Clip z to [-z_thresh, z_thresh]
    - Reconstruct values: median + z_clipped * MAD
    Returns a new DataFrame with same index/columns.
    """
    arr = df_num.values.astype(float)  # shape (n_samples, n_features)
    med = np.nanmedian(arr, axis=0)    # shape (n_features,)
    mads = np.nanmedian(np.abs(arr - med), axis=0)
    mads = np.where(mads == 0, 1.0, mads)  # avoid divide-by-zero
    z = (arr - med) / mads
    z_clipped = np.clip(z, -z_thresh, z_thresh)
    arr2 = med + z_clipped * mads
    return pd.DataFrame(arr2, columns=df_num.columns, index=df_num.index)

# -------------------------
# Main
# -------------------------
def main():
    # folders
    ensure_dir("data")
    ensure_dir("data/processed")
    ensure_dir("models")
    # locate CSV
    candidates = [
        "data/train_data.csv",
        "data/Life Expectancy.csv",
        "data/Life_Expectancy.csv",
        "Life Expectancy.csv",
        "train_data.csv",
        "data/train_data_normalized.csv",
    ]
    csv_path = None
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError("Life Expectancy CSV not found in data/ or repo root. Place it at data/train_data.csv")
    print("Loading:", csv_path)
    df = pd.read_csv(csv_path)

    # Normalize column names
    df = normalize_column_names(df)

    # keep original colnames map for target detection
    # Accept both 'life_expectancy' or similar:
    target_candidates = [c for c in df.columns if "life" in c and "expect" in c]
    if len(target_candidates) == 0:
        raise ValueError("Could not find target column 'life expectancy' in csv headers.")
    target_col = target_candidates[0]
    print("Detected target column:", target_col)

    # 1) Country-wise forward/backfill for key time dependent variables
    time_cols = ["polio", "diphtheria", "hepatitis_b", "gdp", "population", "alcohol", "percentage_expenditure", "total_expenditure"]
    for col in time_cols:
        if col in df.columns:
            try:
                df[col] = fill_country_series(df, col)
            except Exception:
                pass

    # 2) Convert obvious zero-values to NaN for fields where zero is invalid
    zero_to_nan = ["income_composition_of_resources", "percentage_expenditure"]
    for col in zero_to_nan:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    # 3) Replace isolated zeros for schooling if surrounded by non-zero
    if "schooling" in df.columns and "country" in df.columns:
        mask_iso_zero = (df["schooling"] == 0) & (
            df.groupby("country")["schooling"].shift().gt(0) |
            df.groupby("country")["schooling"].shift(-1).gt(0)
        )
        df.loc[mask_iso_zero, "schooling"] = np.nan

    # 4) Impute: try groupby(status, year) medians for gdp/pop/total_expenditure where possible
    if set(["status", "year"]).issubset(df.columns):
        for col in ["gdp", "population", "total_expenditure"]:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df.groupby(["status", "year"])[col].transform("median"))

    # 5) For remaining numeric columns: forward/backfill done, now fallback to group median (status) or global median
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_all = [c for c in numeric_cols_all if c != target_col]
    for col in numeric_cols_all:
        if df[col].isnull().any():
            if "status" in df.columns:
                df[col] = df[col].fillna(df.groupby("status")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())

    # 6) Drop rows missing target
    before = df.shape[0]
    df = df.dropna(subset=[target_col])
    after = df.shape[0]
    print(f"Dropped {before - after} rows with missing target. Remaining rows: {after}")

    # 7) Domain-specific transforms: log1p for skewed count-like cols
    for col in ["measles", "infant_deaths", "under-five_deaths"]:
        if col in df.columns:
            df[col] = np.log1p(df[col].fillna(0).astype(float))

    # 8) BMI winsorize / sensible bounds
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())
        df["bmi"] = df["bmi"].clip(lower=12, upper=60)

    # 9) Feature engineering: Year -> years_since_2000
    if "year" in df.columns:
        df["years_since_2000"] = df["year"].astype(float) - 2000.0

    # 10) Drop country (high-cardinality)
    if "country" in df.columns:
        df = df.drop(columns=["country"])

    # 11) Encode Status as binary (0/1) if present
    if "status" in df.columns:
        unique_status = df["status"].dropna().unique()
        mapping = {}
        if "developed" in [s.lower() for s in unique_status]:
            mapping = {s: (1 if s.lower()=="developed" else 0) for s in unique_status}
        else:
            mapping = {s: (0 if i==0 else 1) for i,s in enumerate(unique_status)}
        df["status_encoded"] = df["status"].map(mapping).fillna(0).astype(int)
        df = df.drop(columns=["status"])

    # 12) Prepare feature matrix X and y
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col])

    # 13) Remove any remaining non-numeric columns (safety)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print("Dropping non-numeric columns:", non_numeric)
        X = X.drop(columns=non_numeric)

    # 14) Outlier handling: cap by z-score using median & MAD
    X_num = X.copy()
    X_num = cap_outliers_zscore(X_num, z_thresh=5.0)

    # 15) Multicollinearity: remove features with |corr| > 0.95
    corr = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if len(to_drop) > 0:
        print("Removing highly correlated features:", to_drop)
    X_num = X_num.drop(columns=to_drop)

    # 16) Final feature set
    feature_names = X_num.columns.tolist()
    print("Final features:", feature_names)

    # 17) Prepare arrays and deterministic split
    X_arr = X_num.values.astype(float)
    rng = np.random.RandomState(42)
    n = X_arr.shape[0]
    perm = rng.permutation(n)
    test_size = int(0.2 * n)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    X_train = X_arr[train_idx]
    X_val = X_arr[test_idx]
    y_train = y[train_idx]
    y_val = y[test_idx]

    # 18) Robust scaling (median & MAD) computed on training set
    X_train_scaled, med, mads = robust_scale_train(X_train)
    X_val_scaled = robust_scale_apply(X_val, med, mads)

    # 19) clip scaled values for numerical stability
    np.clip(X_train_scaled, -5.0, 5.0, out=X_train_scaled)
    np.clip(X_val_scaled, -5.0, 5.0, out=X_val_scaled)

    # 20) Save arrays expected by training pipeline
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)

    np.save("data/processed/X_train_scaled.npy", X_train_scaled)
    np.save("data/processed/X_val_scaled.npy", X_val_scaled)

    # 21) Save cleaned CSV and normalized CSV for compatibility
    # Build normalized full dataframe using med/mads transform
    df_full_norm = pd.DataFrame((X_arr - med) / mads, columns=feature_names)
    df_full_norm = np.clip(df_full_norm, -5.0, 5.0)
    df_full_norm[target_col] = y
    df_full_norm.to_csv("data/train_data_normalized.csv", index=False)

    # save clean (pre-normalized) CSV
    df_clean_pre = pd.DataFrame(X_arr, columns=feature_names)
    df_clean_pre[target_col] = y
    df_clean_pre.to_csv("data/train_data_clean.csv", index=False)

    # 22) Save preprocessor metadata (median, mad, features, removed)
    preprocessor = {
        "feature_names": feature_names,
        "median": med.tolist(),
        "mad": mads.tolist(),
        "removed_features": to_drop,
        "mapping_status": mapping if "mapping" in locals() else {},
        "target_col": target_col
    }
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # 23) Save metadata
    metadata = {
        "n_rows": n,
        "n_features": len(feature_names),
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0]
    }
    with open("data/processed/metadata_pp1y.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Saved arrays and preprocessor. Shapes (train/val):", X_train.shape, X_val.shape)
    print("Saved: data/train_data_clean.csv, data/train_data_normalized.csv, data/processed/*.npy, models/preprocessor.pkl")

if __name__ == "__main__":
    main()

# -------------------------
# Compatibility metadata writer
# -------------------------
# Write a metadata.pkl file tailored to train_model.py expectations.
# This keeps backwards compatibility with training scripts that look for data/processed/metadata.pkl
try:
    compat_path = "data/processed/metadata.pkl"
    # If a metadata_pp1y.pkl exists, load and normalize keys
    src_meta = "data/processed/metadata_pp1y.pkl"
    if os.path.exists(src_meta):
        with open(src_meta, "rb") as f:
            raw_meta = pickle.load(f)
        compat_meta = {
            "feature_names": raw_meta.get("feature_names", raw_meta.get("feature_names_full", [])),
            "n_features": int(raw_meta.get("n_features", len(raw_meta.get("feature_names", [])))),
            "n_rows": int(raw_meta.get("n_rows", 0)),
            "n_train": int(raw_meta.get("n_train", 0)),
            "n_val": int(raw_meta.get("n_val", 0)),
            "target_col": raw_meta.get("target_col", None)
        }
    else:
        # Build from variables available in this script
        compat_meta = {
            "feature_names": feature_names if 'feature_names' in globals() else [],
            "n_features": len(feature_names) if 'feature_names' in globals() else 0,
            "n_rows": int(n) if 'n' in globals() else (int(X_train.shape[0] + X_val.shape[0]) if 'X_train' in globals() and 'X_val' in globals() else 0),
            "n_train": int(X_train.shape[0]) if 'X_train' in globals() else 0,
            "n_val": int(X_val.shape[0]) if 'X_val' in globals() else 0,
            "target_col": target_col if 'target_col' in globals() else None
        }
    # save compatibility metadata
    with open(compat_path, "wb") as f:
        pickle.dump(compat_meta, f)
    print("Wrote compatibility metadata:", compat_path)
except Exception as e:
    print("Warning: failed to write compatibility metadata.pkl:", e)

