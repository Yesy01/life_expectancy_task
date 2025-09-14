"""
Train Model Module - Introduction to Machine Learning
Implements Linear, Polynomial, Ridge, and Lasso Regression from scratch
Demonstrates concepts from IML Lectures 1-6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from typing import Tuple, Dict, List, Optional


class BaseRegression:
    """
    Base class for regression models
    Demonstrates object-oriented design and code reusability
    """
    
    def __init__(self, name: str = "BaseRegression"):
        self.name = name
        self.weights = None
        self.training_history = {}
        self.is_fitted = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using learned weights"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        return X @ self.weights
    
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error - connects to MLE under Gaussian noise assumption"""
        return np.mean((y_true - y_pred) ** 2)
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(self.mse(y_true, y_pred))
    
    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of determination R²"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term (bias) to feature matrix"""
        n_samples = X.shape[0]
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        return X_with_intercept


class LinearRegression(BaseRegression):
    """
    Linear Regression with both closed-form and gradient descent solutions
    Demonstrates understanding of optimization methods from Lectures 4-5
    """
    
    def __init__(self, method: str = 'normal', learning_rate: float = 0.01, 
                 max_iters: int = 1000, tol: float = 1e-6):
        """
        Args:
            method: 'normal' for closed-form, 'gradient' for gradient descent
            learning_rate: Step size for gradient descent (α in slides)
            max_iters: Maximum iterations for gradient descent
            tol: Convergence tolerance
        """
        super().__init__(name="LinearRegression")
        self.method = method
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.cost_history = []
    
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Closed-form solution: θ = (X^T X)^(-1) X^T y
        
        Mathematical justification (MLE perspective):
        - Assumes y = Xθ + ε where ε ~ N(0, σ²)
        - MLE solution equals least squares solution
        - Derived by setting ∇J(θ) = 0
        """
        # Add intercept
        X = self.add_intercept(X)
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        # Using np.linalg.solve for numerical stability instead of matrix inverse
        XtX = X.T @ X
        Xty = X.T @ y
        
        try:
            # More stable than np.linalg.inv(XtX) @ Xty
            self.weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # If singular matrix, use pseudoinverse (Moore-Penrose)
            print("Warning: Singular matrix detected, using pseudoinverse")
            self.weights = np.linalg.pinv(X) @ y
        
        self.is_fitted = True
    
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, 
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None) -> None:
        """
        Gradient Descent optimization
        
        Cost function: J(θ) = (1/2m) Σ(h(x_i) - y_i)²
        Gradient: ∇J(θ) = (1/m) X^T(Xθ - y)
        Update rule: θ := θ - α∇J(θ)
        """
        # Add intercept
        X = self.add_intercept(X)
        if X_val is not None:
            X_val = self.add_intercept(X_val)
        
        n_samples, n_features = X.shape
        
        # Initialize weights (small random values)
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        
        # Track convergence
        self.cost_history = []
        self.val_cost_history = []
        
        for iteration in range(self.max_iters):
            # Forward pass: predictions
            y_pred = X @ self.weights
            
            # Compute cost J(θ)
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Validation cost if provided
            if X_val is not None and y_val is not None:
                y_val_pred = X_val @ self.weights
                val_cost = (1 / (2 * len(y_val))) * np.sum((y_val_pred - y_val) ** 2)
                self.val_cost_history.append(val_cost)
            
            # Compute gradient: ∇J(θ) = (1/m) X^T(Xθ - y)
            gradient = (1 / n_samples) * X.T @ (y_pred - y)
            
            # Update weights: θ := θ - α∇J(θ)
            self.weights = self.weights - self.learning_rate * gradient
            
            # Check convergence
            if iteration > 0:
                cost_change = abs(self.cost_history[-1] - self.cost_history[-2])
                if cost_change < self.tol:
                    print(f"Converged at iteration {iteration}")
                    break
        
        self.is_fitted = True
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'LinearRegression':
        """Fit the model using specified method"""
        if self.method == 'normal':
            self.fit_normal_equation(X, y)
        elif self.method == 'gradient':
            self.fit_gradient_descent(X, y, X_val, y_val)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def get_coefficients(self) -> Tuple[float, np.ndarray]:
        """Return intercept and coefficients separately"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.weights[0], self.weights[1:]


class PolynomialRegression(BaseRegression):
    """
    Polynomial Regression - demonstrates feature engineering and overfitting
    Key concept: Model complexity vs generalization (bias-variance tradeoff)
    """
    
    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        """
        Args:
            degree: Polynomial degree
            include_bias: Whether to include intercept
            interaction_only: If True, only interaction terms (no powers)
        """
        super().__init__(name=f"PolynomialRegression(degree={degree})")
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.linear_reg = LinearRegression(method='normal')
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate polynomial features
        Example for degree=2: [x1, x2] -> [1, x1, x2, x1², x1*x2, x2²]
        """
        n_samples, n_features = X.shape
        
        if self.degree == 1:
            return X
        
        # Generate polynomial features
        features = [X]
        
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                # Only interaction terms
                for i in range(n_features):
                    for j in range(i, n_features):
                        if i != j:
                            features.append((X[:, i] * X[:, j]).reshape(-1, 1))
            else:
                # All polynomial terms
                for i in range(n_features):
                    features.append((X[:, i] ** d).reshape(-1, 1))
                
                # Interaction terms for this degree
                if d == 2:  # Only 2-way interactions for simplicity
                    for i in range(n_features):
                        for j in range(i + 1, n_features):
                            features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        X_poly = np.hstack(features)
        return X_poly
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        """Fit polynomial regression"""
        X_poly = self.transform_features(X)
        self.linear_reg.fit(X_poly, y)
        self.weights = self.linear_reg.weights
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using polynomial features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X_poly = self.transform_features(X)
        X_poly = self.linear_reg.add_intercept(X_poly)
        return X_poly @ self.weights


class RidgeRegression(BaseRegression):
    """
    Ridge Regression (L2 Regularization)
    Demonstrates regularization to prevent overfitting
    
    Cost function: J(θ) = MSE + λ||θ||²
    Closed form: θ = (X^T X + λI)^(-1) X^T y
    """
    
    def __init__(self, lambda_reg: float = 1.0, method: str = 'normal'):
        """
        Args:
            lambda_reg: Regularization strength (λ in slides)
            method: 'normal' or 'gradient'
        """
        super().__init__(name=f"RidgeRegression(λ={lambda_reg})")
        self.lambda_reg = lambda_reg
        self.method = method
    
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ridge closed-form solution: θ = (X^T X + λI)^(-1) X^T y
        Note: Don't regularize intercept term
        """
        X = self.add_intercept(X)
        n_features = X.shape[1]
        
        # Create regularization matrix (don't regularize intercept)
        reg_matrix = self.lambda_reg * np.eye(n_features)
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Ridge solution
        XtX = X.T @ X
        Xty = X.T @ y
        
        self.weights = np.linalg.solve(XtX + reg_matrix, Xty)
        self.is_fitted = True
    
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iters: int = 1000,
                           tol: float = 1e-6) -> None:
        """
        Ridge gradient descent
        Gradient includes regularization term: ∇J(θ) = (1/m)X^T(Xθ - y) + (λ/m)θ
        """
        X = self.add_intercept(X)
        n_samples, n_features = X.shape
        
        # Initialize weights
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        
        self.cost_history = []
        
        for iteration in range(max_iters):
            # Predictions
            y_pred = X @ self.weights
            
            # Cost with regularization
            mse_cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            # Don't include intercept in regularization
            reg_cost = (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights[1:] ** 2)
            total_cost = mse_cost + reg_cost
            self.cost_history.append(total_cost)
            
            # Gradient with regularization
            gradient = (1 / n_samples) * X.T @ (y_pred - y)
            # Add regularization gradient (except for intercept)
            reg_gradient = np.zeros_like(self.weights)
            reg_gradient[1:] = (self.lambda_reg / n_samples) * self.weights[1:]
            gradient += reg_gradient
            
            # Update
            self.weights -= learning_rate * gradient
            
            # Convergence check
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < tol:
                print(f"Ridge converged at iteration {iteration}")
                break
        
        self.is_fitted = True
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """Fit Ridge regression"""
        if self.method == 'normal':
            self.fit_normal_equation(X, y)
        else:
            self.fit_gradient_descent(X, y)
        return self


class LassoRegression(BaseRegression):
    """
    Lasso Regression (L1 Regularization)
    Promotes sparsity in coefficients (feature selection)
    
    Cost function: J(θ) = MSE + λ||θ||₁
    No closed-form solution - requires iterative optimization
    """
    
    def __init__(self, lambda_reg: float = 1.0, learning_rate: float = 0.01,
                 max_iters: int = 1000, tol: float = 1e-6):
        """
        Args:
            lambda_reg: Regularization strength
            learning_rate: Step size for gradient descent
            max_iters: Maximum iterations
            tol: Convergence tolerance
        """
        super().__init__(name=f"LassoRegression(λ={lambda_reg})")
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
    
    def soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft thresholding operator for L1 regularization
        S(x, λ) = sign(x) * max(|x| - λ, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegression':
        """
        Fit Lasso using proximal gradient descent
        Uses soft thresholding for L1 penalty
        """
        X = self.add_intercept(X)
        n_samples, n_features = X.shape
        
        # Initialize weights
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        
        self.cost_history = []
        
        for iteration in range(self.max_iters):
            # Predictions
            y_pred = X @ self.weights
            
            # Cost with L1 regularization
            mse_cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            l1_cost = (self.lambda_reg / n_samples) * np.sum(np.abs(self.weights[1:]))
            total_cost = mse_cost + l1_cost
            self.cost_history.append(total_cost)
            
            # Gradient of MSE term only
            gradient_mse = (1 / n_samples) * X.T @ (y_pred - y)
            
            # Update with gradient step
            self.weights -= self.learning_rate * gradient_mse
            
            # Apply soft thresholding for L1 (except intercept)
            threshold = self.learning_rate * self.lambda_reg / n_samples
            self.weights[1:] = self.soft_threshold(self.weights[1:], threshold)
            
            # Convergence check
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tol:
                print(f"Lasso converged at iteration {iteration}")
                break
        
        self.is_fitted = True
        return self
    
    def get_selected_features(self, feature_names: List[str], 
                            threshold: float = 1e-4) -> List[str]:
        """
        Return features with non-zero coefficients (feature selection property of Lasso)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        selected_indices = np.where(np.abs(self.weights[1:]) > threshold)[0]
        return [feature_names[i] for i in selected_indices]


class ModelComparison:
    """
    Compare different models and demonstrate bias-variance tradeoff
    Key concepts from lectures: model selection, cross-validation, learning curves
    """
    
    def __init__(self, models: Dict[str, BaseRegression]):
        """
        Args:
            models: Dictionary of model_name -> model_instance
        """
        self.models = models
        self.results = {}
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       k_folds: int = 5) -> Dict:
        """
        K-fold cross-validation for all models
        Demonstrates generalization performance estimation
        """
        n_samples = X.shape[0]
        fold_size = n_samples // k_folds
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        cv_results = {name: {'train_scores': [], 'val_scores': []} 
                     for name in self.models.keys()}
        
        for fold in range(k_folds):
            # Create fold indices
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_samples
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            # Split data
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            # Train and evaluate each model
            for name, model in self.models.items():
                # Clone and train model
                if isinstance(model, LinearRegression):
                    model_clone = LinearRegression(method=model.method)
                elif isinstance(model, PolynomialRegression):
                    model_clone = PolynomialRegression(degree=model.degree)
                elif isinstance(model, RidgeRegression):
                    model_clone = RidgeRegression(lambda_reg=model.lambda_reg)
                elif isinstance(model, LassoRegression):
                    model_clone = LassoRegression(lambda_reg=model.lambda_reg)
                else:
                    continue
                
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                train_pred = model_clone.predict(X_train_fold)
                val_pred = model_clone.predict(X_val_fold)
                
                train_rmse = model_clone.rmse(y_train_fold, train_pred)
                val_rmse = model_clone.rmse(y_val_fold, val_pred)
                
                cv_results[name]['train_scores'].append(train_rmse)
                cv_results[name]['val_scores'].append(val_rmse)
        
        # Calculate mean and std
        for name in cv_results:
            cv_results[name]['train_mean'] = np.mean(cv_results[name]['train_scores'])
            cv_results[name]['train_std'] = np.std(cv_results[name]['train_scores'])
            cv_results[name]['val_mean'] = np.mean(cv_results[name]['val_scores'])
            cv_results[name]['val_std'] = np.std(cv_results[name]['val_scores'])
        
        return cv_results
    
    def plot_learning_curves(self, X: np.ndarray, y: np.ndarray,
                           train_sizes: List[float] = None) -> None:
        """
        Plot learning curves to show bias-variance tradeoff
        High bias: both train and val error high
        High variance: low train error, high val error
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= 4:
                break
            
            train_scores = []
            val_scores = []
            
            for train_size in train_sizes:
                n_train = int(train_size * len(X))
                
                # Use first n_train samples for training
                X_train_subset = X[:n_train]
                y_train_subset = y[:n_train]
                
                # Use remaining for validation
                X_val_subset = X[n_train:]
                y_val_subset = y[n_train:]
                
                if len(X_val_subset) == 0:
                    X_val_subset = X_train_subset
                    y_val_subset = y_train_subset
                
                # Clone and train model
                if isinstance(model, LinearRegression):
                    model_clone = LinearRegression(method=model.method)
                elif isinstance(model, PolynomialRegression):
                    model_clone = PolynomialRegression(degree=model.degree)
                elif isinstance(model, RidgeRegression):
                    model_clone = RidgeRegression(lambda_reg=model.lambda_reg)
                elif isinstance(model, LassoRegression):
                    model_clone = LassoRegression(lambda_reg=model.lambda_reg)
                
                model_clone.fit(X_train_subset, y_train_subset)
                
                # Evaluate
                train_pred = model_clone.predict(X_train_subset)
                val_pred = model_clone.predict(X_val_subset)
                
                train_scores.append(model_clone.rmse(y_train_subset, train_pred))
                val_scores.append(model_clone.rmse(y_val_subset, val_pred))
            
            # Plot
            actual_sizes = [int(size * len(X)) for size in train_sizes]
            axes[idx].plot(actual_sizes, train_scores, 'o-', label='Training error')
            axes[idx].plot(actual_sizes, val_scores, 'o-', label='Validation error')
            axes[idx].set_xlabel('Training Set Size')
            axes[idx].set_ylabel('RMSE')
            axes[idx].set_title(f'Learning Curve: {name}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/learning_curves.png')
        plt.show()


def main():
    """
    Main training pipeline
    Demonstrates complete workflow from data loading to model comparison
    """
    print("="*60)
    print("TRAINING REGRESSION MODELS - IML ASSIGNMENT")
    print("="*60)
    
    # Load preprocessed data
    print("\n1. Loading preprocessed data...")
    X_train = np.load('data/processed/X_train_scaled.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val_scaled.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    with open('data/processed/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Training samples: {X_train.shape}")
    print(f"Validation samples: {X_val.shape}")
    print(f"Features: {metadata['n_features']}")
    
    # Initialize models
    print("\n2. Initializing models...")
    models = {
        'Linear (Normal Eq)': LinearRegression(method='normal'),
        'Linear (Gradient)': LinearRegression(method='gradient', learning_rate=0.1),
        'Polynomial (deg=2)': PolynomialRegression(degree=2),
        'Polynomial (deg=3)': PolynomialRegression(degree=3),
        'Ridge (λ=0.1)': RidgeRegression(lambda_reg=0.1),
        'Ridge (λ=1.0)': RidgeRegression(lambda_reg=1.0),
        'Ridge (λ=10)': RidgeRegression(lambda_reg=10.0),
        'Lasso (λ=0.01)': LassoRegression(lambda_reg=0.01),
        'Lasso (λ=0.1)': LassoRegression(lambda_reg=0.1),
    }
    
    # Train and evaluate each model
    print("\n3. Training models...")
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train
        if 'Gradient' in name:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        results[name] = {
            'train_mse': model.mse(y_train, y_train_pred),
            'train_rmse': model.rmse(y_train, y_train_pred),
            'train_r2': model.r_squared(y_train, y_train_pred),
            'val_mse': model.mse(y_val, y_val_pred),
            'val_rmse': model.rmse(y_val, y_val_pred),
            'val_r2': model.r_squared(y_val, y_val_pred),
            'training_time': time.time() - start_time
        }
        
        # Save model
        model_filename = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        with open(f'models/{model_filename}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  Train RMSE: {results[name]['train_rmse']:.4f}")
        print(f"  Val RMSE: {results[name]['val_rmse']:.4f}")
        print(f"  Val R²: {results[name]['val_r2']:.4f}")
    
    # Cross-validation
    print("\n4. Performing 5-fold cross-validation...")
    comparator = ModelComparison(models)
    cv_results = comparator.cross_validate(X_train, y_train, k_folds=5)
    
    # Display results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"{'Model':<25} {'CV Train RMSE':<20} {'CV Val RMSE':<20}")
    print("-"*65)
    
    for name in models.keys():
        train_str = f"{cv_results[name]['train_mean']:.4f} ± {cv_results[name]['train_std']:.4f}"
        val_str = f"{cv_results[name]['val_mean']:.4f} ± {cv_results[name]['val_std']:.4f}"
        print(f"{name:<25} {train_str:<20} {val_str:<20}")
    
    # Save best model
    best_model_name = min(cv_results.keys(), key=lambda k: cv_results[k]['val_mean'])
    print(f"\n Best model: {best_model_name}")
    
    # Save as final model
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)  # Retrain on full training set
    with open('models/regression_model_final.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save results
    with open('results/training_results.pkl', 'wb') as f:
        pickle.dump({
            'model_results': results,
            'cv_results': cv_results,
            'best_model': best_model_name
        }, f)
    
    print("\nTraining complete! Models saved to models/")
    print("Results saved to results/")
    
    # Generate learning curves
    print("\n5. Generating learning curves...")
    comparator.plot_learning_curves(X_train, y_train)
    
    return results, cv_results


if __name__ == "__main__":
    results, cv_results = main()