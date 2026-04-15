"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from typing import Tuple, Optional, List


class DataPreprocessor:
    """Preprocess data for machine learning."""
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.label_encoders = {}
        self.numeric_features = []
        self.categorical_features = []
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Training features
            
        Returns:
            Self
        """
        # Identify numeric and categorical features
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        # Fit scaler on numeric features
        if self.numeric_features:
            self.scaler.fit(X[self.numeric_features])
        
        # Fit label encoders on categorical features
        for col in self.categorical_features:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            oe.fit(X[[col]].astype(str))
            self.label_encoders[col] = oe
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        X_transformed = X.copy()
        
        # Scale numeric features
        if self.numeric_features and self.scaler is not None:
            X_transformed[self.numeric_features] = self.scaler.transform(
                X[self.numeric_features]
            )
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in self.label_encoders:
                X_transformed[col] = self.label_encoders[col].transform(
                    X[[col]].astype(str)
                ).flatten()
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform features.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed features
        """
        return self.fit(X).transform(X)
    
    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            X: Features with potential missing values
            strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        X_clean = X.copy()
        
        if strategy == 'drop':
            X_clean = X_clean.dropna()
        else:
            for col in X_clean.columns:
                if X_clean[col].isnull().any():
                    if col in self.numeric_features:
                        if strategy == 'mean':
                            X_clean[col].fillna(X_clean[col].mean(), inplace=True)
                        elif strategy == 'median':
                            X_clean[col].fillna(X_clean[col].median(), inplace=True)
                    else:
                        # For categorical, use mode
                        X_clean[col].fillna(X_clean[col].mode()[0], inplace=True)
        
        return X_clean
    
    def remove_outliers(
        self,
        X: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from numeric features.
        
        Args:
            X: Features
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        X_clean = X.copy()
        
        for col in self.numeric_features:
            if col in X_clean.columns:
                if method == 'iqr':
                    Q1 = X_clean[col].quantile(0.25)
                    Q3 = X_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    X_clean = X_clean[
                        (X_clean[col] >= lower_bound) & 
                        (X_clean[col] <= upper_bound)
                    ]
                elif method == 'zscore':
                    z_scores = np.abs((X_clean[col] - X_clean[col].mean()) / X_clean[col].std())
                    X_clean = X_clean[z_scores < threshold]
        
        return X_clean
