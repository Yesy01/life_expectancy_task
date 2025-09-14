import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path to import model classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model classes so pickle can load them
try:
    from train_model import (
        RobustLinearRegression,
        RobustPolynomialRegression,
        RobustRidgeRegression,
        RobustLassoRegression
    )
    from data_preprocessing import RobustDataPreprocessor
except ImportError:
    print("Warning: Could not import model classes")


def load_model(model_path):
    """Load saved model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_and_preprocess_data(data_path, preprocessor_path='models/preprocessor.pkl'):
    """Load and preprocess test data using saved preprocessor or fallback"""
    try:
        # Load preprocessor
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        else:
            preprocessor = None

        # Case 1: DataPreprocessor object saved
        if preprocessor is not None and hasattr(preprocessor, "preprocess_pipeline"):
            data = preprocessor.preprocess_pipeline(data_path)
            X = data['X_train_scaled']
            y = data['y_train']
            return X, y

        # Case 2: dict metadata (from data_pp1y.py compatibility mode)
        elif isinstance(preprocessor, dict):
            print("Info: Preprocessor is metadata dict, loading processed arrays directly:")
            X = np.load("data/processed/X_train_scaled.npy")
            y = np.load("data/processed/y_train.npy")
            return X, y

        # Case 3: No usable preprocessor found → fallback
        else:
            print("Warning: No valid preprocessor found, using basic preprocessing")
            df = pd.read_csv(data_path)

            if 'Life expectancy ' in df.columns:
                df = df.dropna(subset=['Life expectancy '])
                y = df['Life expectancy '].values
                X = df.select_dtypes(include=[np.number]).drop('Life expectancy ', axis=1).values
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                X = np.clip(X, -5, 5)
                return X, y
            else:
                raise ValueError("Target column 'Life expectancy ' not found")

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise



def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    # Ensure arrays are 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return mse, rmse, r2


def save_metrics(metrics_output_path, mse, rmse, r2):
    """Save metrics in exact assignment format"""
    metrics_content = f"""Regression Metrics:
Mean Squared Error (MSE): {mse:.2f}
Root Mean Squared Error (RMSE): {rmse:.2f}
R-squared (R²) Score: {r2:.2f}"""
    
    try:
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            f.write(metrics_content)
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False


def save_predictions(predictions_output_path, predictions):
    """Save predictions as single column CSV with no header"""
    try:
        os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
        # Ensure predictions is 1D array
        predictions = np.array(predictions).flatten()
        # Save as single column, no header
        np.savetxt(predictions_output_path, predictions, fmt='%.6f', delimiter=',')
        return True
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return False


def main():
    """Main evaluation function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate regression model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--metrics_output_path', type=str, required=True,
                       help='Path where evaluation metrics will be saved')
    parser.add_argument('--predictions_output_path', type=str, required=True,
                       help='Path where predictions will be saved')
    
    args = parser.parse_args()
    
    print("MODEL EVALUATION")
    
    # Load model
    print("\n1. Loading model:")
    model = load_model(args.model_path)
    if model is None:
        print(" Failed to load model")
        return
    print(f"   Model loaded: {type(model).__name__}")
    
    # Load and preprocess data
    print("\n2. Loading and preprocessing data:")
    try:
        X, y = load_and_preprocess_data(args.data_path)
        print(f"   Data shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
    except Exception as e:
        print(f" Failed to load data: {e}")
        return
    
    # Make predictions
    print("\n3. Making predictions:")
    try:
        predictions = model.predict(X)
        predictions = np.array(predictions).flatten()
        print(f"   Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f" Failed to make predictions: {e}")
        return
    
    # Calculate metrics
    print("\n4. Calculating metrics:")
    try:
        mse, rmse, r2 = calculate_metrics(y, predictions)
        print(f"   MSE: {mse:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R²: {r2:.2f}")
    except Exception as e:
        print(f" Failed to calculate metrics: {e}")
        return
    
    # Save results
    print("\n5. Saving results:")
    metrics_saved = save_metrics(args.metrics_output_path, mse, rmse, r2)
    predictions_saved = save_predictions(args.predictions_output_path, predictions)
    
    if metrics_saved and predictions_saved:
        print(f"    Metrics saved to {args.metrics_output_path}")
        print(f"    Predictions saved to {args.predictions_output_path}")
        print("\n Evaluation complete!")
    else:
        print("\ Some files may not have been saved correctly")


if __name__ == "__main__":
    main()