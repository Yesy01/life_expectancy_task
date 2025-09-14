

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class RobustDataPreprocessor:
   
    
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.feature_names = None
        self.is_fitted = False
        self.problematic_features = []
    
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def handle_missing_values(self, df):
        df_imputed = df.copy()
        if 'Life expectancy ' in df_imputed.columns:
            df_imputed = df_imputed.dropna(subset=['Life expectancy '])
        if 'Status' in df_imputed.columns:
            for status in df_imputed['Status'].unique():
                mask = df_imputed['Status'] == status
                num_cols = df_imputed.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    if df_imputed[col].isnull().any():
                        med = df_imputed.loc[mask, col].median()
                        if pd.isna(med):
                            med = df_imputed[col].median()
                        df_imputed.loc[mask & df_imputed[col].isnull(), col] = med
        else:
            num_cols = df_imputed.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if df_imputed[col].isnull().any():
                    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        return df_imputed
    
    def remove_outliers(self, df, columns_to_check=None, z_threshold=5):
        df_clean = df.copy()
        if columns_to_check is None:
            columns_to_check = df_clean.select_dtypes(include=[np.number]).columns
        for col in columns_to_check:
            if col == 'Life expectancy ':
                continue
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                z = np.abs((df_clean[col] - mean) / std)
                mask = z > z_threshold
                if mask.any():
                    upper = mean + z_threshold * std
                    lower = mean - z_threshold * std
                    df_clean[col] = df_clean[col].clip(lower, upper)
        return df_clean
    
    def encode_categorical(self, df):
        df_enc = df.copy()
        if 'Country' in df_enc.columns:
            df_enc = df_enc.drop('Country', axis=1)
        if 'Status' in df_enc.columns:
            df_enc['status_encoded'] = (df_enc['Status'] == 'Developed').astype(int)
            df_enc = df_enc.drop('Status', axis=1)
        return df_enc
    
    def engineer_features(self, df):
        df_feat = df.copy()
        if 'Year' in df_feat.columns:
            df_feat['years_since_2000'] = df_feat['Year'] - 2000
            df_feat = df_feat.drop('Year', axis=1)
        return df_feat
    
    def robust_scale(self, X):
        X_scaled = X.copy()
        for i in range(X_scaled.shape[1]):
            col = X_scaled[:, i]
            median = np.median(col)
            mad = np.median(np.abs(col - median))
            if mad > 0:
                X_scaled[:, i] = (col - median) / (1.4826 * mad)
            else:
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    X_scaled[:, i] = (col - col_min) / (col_max - col_min)
                else:
                    X_scaled[:, i] = 0
        return np.clip(X_scaled, -5, 5)
    
    def fit(self, X):
        self.feature_medians = np.median(X, axis=0)
        self.feature_mads = np.median(np.abs(X - self.feature_medians), axis=0)
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        X_scaled = X.copy()
        for i in range(X_scaled.shape[1]):
            if self.feature_mads[i] > 0:
                X_scaled[:, i] = (X[:, i] - self.feature_medians[i]) / (1.4826 * self.feature_mads[i])
            elif self.feature_stds[i] > 0:
                X_scaled[:, i] = (X[:, i] - self.feature_means[i]) / self.feature_stds[i]
            else:
                X_scaled[:, i] = 0
        return np.clip(X_scaled, -5, 5)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def check_multicollinearity(self, X, feature_names):
        corr = np.corrcoef(X.T)
        high_corr = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if abs(corr[i, j]) > 0.95:
                    high_corr.append((i, j))
                    print(f"   High correlation: {feature_names[i]} - {feature_names[j]} ({corr[i,j]:.3f})")
        to_remove = set(j for i, j in high_corr)
        if to_remove:
            keep = [i for i in range(X.shape[1]) if i not in to_remove]
            return X[:, keep], [feature_names[i] for i in keep]
        return X, feature_names
    
    def train_test_split(self, X, y, test_size=0.2, random_seed=42):
        np.random.seed(random_seed)
        n = X.shape[0]
        n_test = int(n * test_size)
        idx = np.random.permutation(n)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    def preprocess_pipeline(self, filepath, target_column='Life expectancy ', test_size=0.2, random_seed=42):
        print("ROBUST DATA PREPROCESSING PIPELINE")
        
        df = self.load_data(filepath)
        print(f"   Original shape: {df.shape}")
        
        df = self.handle_missing_values(df)
        print(f"   After handling missing target: {df.shape}")
        
        df = self.remove_outliers(df, z_threshold=5)
        df = self.encode_categorical(df)
        df = self.engineer_features(df)
        
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values
        self.feature_names = df.drop(target_column, axis=1).columns.tolist()
        
        X, self.feature_names = self.check_multicollinearity(X, self.feature_names)
        
        X_train, X_val, y_train, y_val = self.train_test_split(X, y, test_size, random_seed)
        X_train_scaled = self.fit_transform(X_train)
        X_val_scaled = self.transform(X_val)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'feature_names': self.feature_names,
            'feature_medians': self.feature_medians,
            'feature_mads': self.feature_mads
        }
    
    def save_preprocessor(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


def main():
    preprocessor = RobustDataPreprocessor()
    data = preprocessor.preprocess_pipeline('data/train_data.csv')
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    np.save('data/processed/X_train.npy', data['X_train'])
    np.save('data/processed/X_val.npy', data['X_val'])
    np.save('data/processed/y_train.npy', data['y_train'])
    np.save('data/processed/y_val.npy', data['y_val'])
    np.save('data/processed/X_train_scaled.npy', data['X_train_scaled'])
    np.save('data/processed/X_val_scaled.npy', data['X_val_scaled'])
    
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    metadata = {
        'feature_names': data['feature_names'],
        'n_features': len(data['feature_names']),
        'n_train': len(data['y_train']),
        'n_val': len(data['y_val'])
    }
    with open('data/processed/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Also save clean & normalized CSVs
    train_norm = pd.DataFrame(data['X_train_scaled'], columns=data['feature_names'])
    train_norm['life_expectancy'] = data['y_train']
    train_norm.to_csv('data/train_data_normalized.csv', index=False)
    
    train_clean = pd.DataFrame(data['X_train'], columns=data['feature_names'])
    train_clean['life_expectancy'] = data['y_train']
    train_clean.to_csv('data/train_data_clean.csv', index=False)
    
    print("\ Preprocessing complete and all files saved!")


if __name__ == "__main__":
    main()
