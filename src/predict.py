"""
Predict Script - Evaluation and Metrics Generation
Follows exact assignment specifications for output format
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import os


def load_model(model_path):
    """Load saved model from pickle file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(data_path):
    """Load and preprocess test data"""
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Apply same preprocessing as training
    # Load preprocessor
    preprocessor_path = os.path.join(os.path.dirname(model_path), 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Preprocess using saved preprocessor
        data = preprocessor.preprocess_pipeline(data_path)
        X = data['X_train_scaled']  # Uses full data as "train"
        y = data['y_train']
    else:
        # Manual preprocessing if preprocessor not found
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline(data_path)
        X = data['X_train_scaled']
        y = data['y_train']
    
    return X, y


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, rmse, r2


def save_metrics(metrics_output_path, mse, rmse, r2):
    """Save metrics in exact assignment format"""
    metrics_content = f"""Regression Metrics:
Mean Squared Error (MSE): {mse:.2f}
Root Mean Squared Error (RMSE): {rmse:.2f}
R-squared (R²) Score: {r2:.2f}"""
    
    with open(metrics_output_path, 'w') as f:
        f.write(metrics_content)


def save_predictions(predictions_output_path, predictions):
    """Save predictions as single column CSV with no header"""
    # Save as single column, no header
    np.savetxt(predictions_output_path, predictions, fmt='%.6f', delimiter=',')


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
    
    # Make model_path available globally for load_data function
    global model_path
    model_path = args.model_path
    
    print("Loading model...")
    model = load_model(args.model_path)
    
    print("Loading and preprocessing data...")
    X, y = load_data(args.data_path)
    
    print("Making predictions...")
    # Handle different model types
    if hasattr(model, 'predict'):
        predictions = model.predict(X)
    else:
        # Fallback for simple weight-based models
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        predictions = X_with_intercept @ model.weights
    
    print("Calculating metrics...")
    mse, rmse, r2 = calculate_metrics(y, predictions)
    
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}")
    
    print("Saving results...")
    save_metrics(args.metrics_output_path, mse, rmse, r2)
    save_predictions(args.predictions_output_path, predictions)
    
    print(f"✅ Metrics saved to {args.metrics_output_path}")
    print(f"✅ Predictions saved to {args.predictions_output_path}")


if __name__ == "__main__":
    main()