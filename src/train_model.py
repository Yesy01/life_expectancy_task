
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')


class RobustLinearRegression:
    """Linear Regression with numerical stability safeguards"""
    
    def __init__(self, regularization=1e-8):
        """
        Args:
            regularization: Small value to add for numerical stability
        """
        self.regularization = regularization
        self.weights = None
        self.is_fitted = False
    
    def add_intercept(self, X):
        """Add intercept term"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def fit(self, X, y):
        """Fit using regularized normal equation"""
        # Add intercept
        X_with_intercept = self.add_intercept(X)
        n_features = X_with_intercept.shape[1]
        
        # Regularized normal equation: θ = (X^T X + λI)^(-1) X^T y
        # This prevents singular matrix issues
        XtX = X_with_intercept.T @ X_with_intercept
        reg_matrix = self.regularization * np.eye(n_features)
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        Xty = X_with_intercept.T @ y
        
        try:
            self.weights = np.linalg.solve(XtX + reg_matrix, Xty)
        except np.linalg.LinAlgError:
            print("Warning: Using pseudoinverse due to singular matrix")
            self.weights = np.linalg.pinv(X_with_intercept) @ y
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X_with_intercept = self.add_intercept(X)
        return X_with_intercept @ self.weights
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


class RobustPolynomialRegression:
    """Polynomial Regression with stability checks"""
    
    def __init__(self, degree=2, regularization=1e-6):
        self.degree = degree
        self.linear_model = RobustLinearRegression(regularization=regularization)
        self.is_fitted = False
    
    def transform_features(self, X):
        """Generate polynomial features with scaling"""
        n_samples, n_features = X.shape
        
        if self.degree == 1:
            return X
        
        # Start with original features
        features = [X]
        
        # Add polynomial terms (be careful with high degrees)
        for d in range(2, min(self.degree + 1, 4)):  # Cap at degree 3 for stability
            for i in range(min(n_features, 10)):  # Limit features to prevent explosion
                poly_feature = X[:, i] ** d
                # Scale down polynomial features to prevent overflow
                poly_feature = poly_feature / (d ** 2)
                features.append(poly_feature.reshape(-1, 1))
        
        # Add selected interaction terms (limit to prevent feature explosion)
        if self.degree >= 2 and n_features > 1:
            for i in range(min(n_features, 5)):
                for j in range(i + 1, min(n_features, 5)):
                    interaction = X[:, i] * X[:, j]
                    features.append(interaction.reshape(-1, 1))
        
        X_poly = np.hstack(features)
        return X_poly
    
    def fit(self, X, y):
        """Fit polynomial regression"""
        X_poly = self.transform_features(X)
        self.linear_model.fit(X_poly, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using polynomial features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X_poly = self.transform_features(X)
        return self.linear_model.predict(X_poly)
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


class RobustRidgeRegression:
    """Ridge Regression (always stable due to regularization)"""
    
    def __init__(self, alpha=1.0):
        self.alpha = max(alpha, 1e-8)  # Ensure minimum regularization
        self.weights = None
        self.is_fitted = False
    
    def add_intercept(self, X):
        """Add intercept term"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def fit(self, X, y):
        """Fit Ridge regression"""
        X_with_intercept = self.add_intercept(X)
        n_features = X_with_intercept.shape[1]
        
        # Ridge solution: θ = (X^T X + αI)^(-1) X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        reg_matrix = self.alpha * np.eye(n_features)
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        Xty = X_with_intercept.T @ y
        
        # This should always work due to regularization
        self.weights = np.linalg.solve(XtX + reg_matrix, Xty)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X_with_intercept = self.add_intercept(X)
        return X_with_intercept @ self.weights
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


class RobustLassoRegression:
    """Lasso Regression using coordinate descent"""
    
    def __init__(self, alpha=1.0, max_iters=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.intercept = 0
        self.is_fitted = False
    
    def soft_threshold(self, rho, alpha):
        """Soft thresholding operator"""
        if rho < -alpha:
            return rho + alpha
        elif rho > alpha:
            return rho - alpha
        else:
            return 0
    
    def fit(self, X, y):
        """Fit Lasso using coordinate descent"""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.intercept = np.mean(y)
        
        # Center y
        y_centered = y - self.intercept
        
        for iteration in range(self.max_iters):
            weights_old = self.weights.copy()
            
            # Coordinate descent
            for j in range(n_features):
                # Compute residual without feature j
                residual = y_centered - X @ self.weights + X[:, j] * self.weights[j]
                
                # Compute rho_j
                rho_j = X[:, j].T @ residual / n_samples
                
                # Apply soft thresholding
                self.weights[j] = self.soft_threshold(rho_j, self.alpha) / (np.sum(X[:, j] ** 2) / n_samples)
            
            # Check convergence
            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                break
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return X @ self.weights + self.intercept
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate a model and return metrics"""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    val_mse = np.mean((y_val - y_val_pred) ** 2)
    
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    
    train_r2 = model.score(X_train, y_train)
    val_r2 = model.score(X_val, y_val)
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }


def main():
    print(" MODEL TRAINING PIPELINE")

    
    # Load preprocessed data
    print("\n1. Loading preprocessed data:")
    try:
        X_train = np.load('data/processed/X_train_scaled.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_val = np.load('data/processed/X_val_scaled.npy')
        y_val = np.load('data/processed/y_val.npy')
        
        with open('data/processed/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"   Training samples: {X_train.shape}")
        print(f"   Validation samples: {X_val.shape}")
        print(f"   Features: {X_train.shape[1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please run data_preprocessing.py first")
        return
    
    # Define models
    print("\n2. Training models:")
    models = {
        'Linear Regression': RobustLinearRegression(regularization=1e-8),
        'Polynomial (deg=2)': RobustPolynomialRegression(degree=2, regularization=1e-6),
        'Ridge (α=0.1)': RobustRidgeRegression(alpha=0.1),
        'Ridge (α=1.0)': RobustRidgeRegression(alpha=1.0),
        'Ridge (α=10.0)': RobustRidgeRegression(alpha=10.0),
        'Lasso (α=0.01)': RobustLassoRegression(alpha=0.01),
        'Lasso (α=0.1)': RobustLassoRegression(alpha=0.1),
    }
    
    results = {}
    best_score = -float('inf')
    best_model = None
    best_name = None
    
    for name, model in models.items():
        print(f"\n   Training {name}:")
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
            metrics['training_time'] = training_time
            results[name] = metrics
            
            print(f"      Train RMSE: {metrics['train_rmse']:.4f}")
            print(f"      Val RMSE: {metrics['val_rmse']:.4f}")
            print(f"      Val R²: {metrics['val_r2']:.4f}")
            
            if metrics['val_r2'] > best_score:
                best_score = metrics['val_r2']
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"      Error training {name}: {e}")
            continue
    
    if best_model is None:
        print("\n No models trained successfully!")
        return
    
    print(f"\n3. Best model: {best_name} (Val R² = {best_score:.4f})")
    
    print("\n4. Saving models:")
    
    model_mapping = {
        'regression_model1.pkl': models.get('Linear Regression', models.get('Ridge (α=1.0)')),
        'regression_model2.pkl': models.get('Polynomial (deg=2)', models.get('Ridge (α=0.1)')),
        'regression_model3.pkl': models.get('Ridge (α=1.0)', models.get('Lasso (α=0.1)')),
        'regression_model_final.pkl': best_model
    }
    
    for filename, model in model_mapping.items():
        if model is not None:
            with open(f'models/{filename}', 'wb') as f:
                pickle.dump(model, f)
                print(f"   Saved: {filename}")
    
    # Save results
    with open('results/training_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'best_model': best_name,
            'best_score': best_score
        }, f)
    
    # Print summary
    print("TRAINING SUMMARY")

    print(f"{'Model':<20} {'Train RMSE':<12} {'Val RMSE':<12} {'Val R²':<10}")
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['train_rmse']:<12.4f} {metrics['val_rmse']:<12.4f} {metrics['val_r2']:<10.4f}")
    
    print("\ Training complete!")
    print(f"   Best model: {best_name} with R² = {best_score:.4f}")
    
    return results, best_model, best_name


if __name__ == "__main__":
    results, best_model, best_name = main()