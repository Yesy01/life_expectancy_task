import pandas as pd
import numpy as np
import pickle
import os


class DataPreprocessor:
    
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.feature_names = None
        self.is_fitted = False
    
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def impute_missing_values(self, df):
        df_imputed = df.copy()
        
        # Separate by Status for grouped imputation (if Status exists)
        if 'Status' in df_imputed.columns:
            for status in df_imputed['Status'].unique():
                status_mask = df_imputed['Status'] == status
                numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
                
                for col in numerical_cols:
                    if df_imputed[col].isnull().any():
                        median_value = df_imputed.loc[status_mask, col].median()
                        if pd.isna(median_value):
                            median_value = df_imputed[col].median()
                        df_imputed.loc[status_mask & df_imputed[col].isnull(), col] = median_value
        else:
            # Simple median imputation if Status doesn't exist
            numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_imputed[col].isnull().any():
                    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        
        return df_imputed
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Encode Status if it exists
        if 'Status' in df_encoded.columns:
            df_encoded['Status_Encoded'] = (df_encoded['Status'] == 'Developed').astype(int)
            df_encoded = df_encoded.drop('Status', axis=1)
        
        return df_encoded
    
    def engineer_features(self, df):
        """Create additional features"""
        df_feat = df.copy()
        
        # Create years_since_2000 if Year exists
        if 'Year' in df_feat.columns:
            df_feat['Years_Since_2000'] = df_feat['Year'] - 2000
            df_feat = df_feat.drop('Year', axis=1)
        
        # Create interaction features if columns exist
        if 'GDP' in df_feat.columns and 'Schooling' in df_feat.columns:
            df_feat['GDP_x_Education'] = df_feat['GDP'] * df_feat['Schooling']
        
        if 'Income composition of resources' in df_feat.columns and 'Schooling' in df_feat.columns:
            df_feat['Income_x_Education'] = df_feat['Income composition of resources'] * df_feat['Schooling']
        
        return df_feat
    
    def fit(self, X):
        """Calculate statistics for scaling"""
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1  # Avoid division by zero
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply scaling using fitted statistics"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        return (X - self.feature_means) / self.feature_stds
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def preprocess_pipeline(self, filepath, target_column='Life expectancy ', test_size=0.2, random_seed=42):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target variable
            test_size: Proportion for validation set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with processed data and metadata
        """
        # Load data
        print("Loading data...")
        df = self.load_data(filepath)
        print(f"Original shape: {df.shape}")
        
        # Drop Country column if it exists
        if 'Country' in df.columns:
            print("Dropping Country column...")
            df = df.drop('Country', axis=1)
        
        # Handle missing values
        print("Imputing missing values...")
        df = self.impute_missing_values(df)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        df = self.encode_categorical(df)
        
        # Engineer features
        print("Engineering features...")
        df = self.engineer_features(df)
        
        # Separate features and target
        print("Separating features and target...")
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values
        self.feature_names = df.drop(target_column, axis=1).columns.tolist()
        
        # Manual train-test split
        print("Splitting data...")
        X_train, X_val, y_train, y_val = self.train_test_split(X, y, test_size, random_seed)
        
        # Fit preprocessor on training data and transform both sets
        print("Scaling features...")
        X_train_scaled = self.fit_transform(X_train)
        X_val_scaled = self.transform(X_val)
        
        print(f"Final shapes - Train: {X_train.shape}, Val: {X_val.shape}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_seed=42):
        """Manually split data into train and test sets"""
        np.random.seed(random_seed)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        # Create shuffled indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split data
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """Save preprocessor object"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_preprocessor(filepath):
        """Load preprocessor object"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def main():
    """
    Main function to run preprocessing
    This will be called when running the script directly
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    data = preprocessor.preprocess_pipeline('data/train_data.csv')
    
    # Save processed data
    print("\nSaving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_train.npy', data['X_train'])
    np.save('data/processed/X_val.npy', data['X_val'])
    np.save('data/processed/y_train.npy', data['y_train'])
    np.save('data/processed/y_val.npy', data['y_val'])
    np.save('data/processed/X_train_scaled.npy', data['X_train_scaled'])
    np.save('data/processed/X_val_scaled.npy', data['X_val_scaled'])
    
    # Save preprocessor object for later use
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Save metadata
    metadata = {
        'feature_names': data['feature_names'],
        'n_features': len(data['feature_names']),
        'n_train': len(data['y_train']),
        'n_val': len(data['y_val'])
    }
    
    with open('data/processed/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n Preprocessing complete!")
    print(f"Training samples: {len(data['y_train'])}")
    print(f"Validation samples: {len(data['y_val'])}")
    print(f"Number of features: {len(data['feature_names'])}")
    print("\nProcessed data saved in data/processed/")
    print("Preprocessor saved in models/preprocessor.pkl")


if __name__ == "__main__":
    main()