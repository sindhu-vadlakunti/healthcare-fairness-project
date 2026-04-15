"""Data Ethics Profiler to analyze inherent bias in datasets."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List


class DataEthicsProfiler:
    """Analyze the 'Inherent Bias' in data before modeling."""
    
    def profile_dataset(self, X: pd.DataFrame, y: pd.Series, sensitive_col: str, save_path: str = None):
        """
        Create a report on how sensitive features correlate with other features and target.
        """
        print(f"Profiling ethical risks for {sensitive_col}...")
        
        # Prepare data (handle categoricals & missing)
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                X_encoded[col] = pd.factorize(X_encoded[col])[0]
        
        X_encoded = X_encoded.fillna(0)
        
        # 1. Mutual Information with Target
        mi = mutual_info_classif(X_encoded, y, discrete_features='auto')
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        
        # 2. Correlation with Sensitive Column
        corr_with_sensitive = X_encoded.corrwith(X_encoded[sensitive_col]).abs().sort_values(ascending=False)
        
        # 3. Disparate Base Rates
        groups = X[sensitive_col].unique()
        base_rates = {}
        for group in groups:
            base_rates[group] = y[X[sensitive_col] == group].mean()
            
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(base_rates.keys()), y=list(base_rates.values()))
        plt.title(f"Historical Base Rates by {sensitive_col}")
        plt.ylabel("Positive Class Frequency")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'mi_ranking': mi_series,
            'sensitive_correlation': corr_with_sensitive,
            'base_rates': base_rates
        }
