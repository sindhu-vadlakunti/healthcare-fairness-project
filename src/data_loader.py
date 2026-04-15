"""Data loading utilities for healthcare datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import urllib.request
from sklearn.model_selection import train_test_split

from src.config_loader import get_config


class DataLoader:
    """Load and prepare healthcare datasets."""
    
    def __init__(self):
        """Initialize data loader."""
        self.config = get_config()
        self.random_seed = self.config.get('project.random_seed', 42)
    
    def load_pima_diabetes(self, download: bool = True) -> pd.DataFrame:
        """
        Load Pima Indians Diabetes dataset.
        
        Args:
            download: Whether to download if not present
            
        Returns:
            DataFrame with diabetes data
        """
        data_dir = self.config.get_data_dir('raw')
        file_path = data_dir / 'pima_diabetes.csv'
        
        if not file_path.exists() and download:
            print("Downloading Pima Indians Diabetes dataset...")
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded to {file_path}")
        
        # Column names for Pima dataset
        columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        df = pd.read_csv(file_path, names=columns)
        
        # Create age groups for sensitive attribute analysis (matching engine bins)
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 60, 100], 
                                labels=['<40', '40-60', '>60'])
        
        return df
    
    def load_adult_income(self, download: bool = True) -> pd.DataFrame:
        """
        Load Adult Income dataset.
        
        Args:
            download: Whether to download if not present
            
        Returns:
            DataFrame with adult income data
        """
        data_dir = self.config.get_data_dir('raw')
        file_path = data_dir / 'adult_income.csv'
        
        if not file_path.exists() and download:
            print("Downloading Adult Income dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded to {file_path}")
        
        # Column names for Adult dataset
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
            'income'
        ]
        
        df = pd.read_csv(file_path, names=columns, skipinitialspace=True)
        
        # Clean income column
        df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1})
        
        # Clean categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        return df
    
    def prepare_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and split dataset into train/test sets.
        
        Args:
            dataset_name: Name of dataset ('pima_diabetes' or 'adult_income')
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load dataset
        if dataset_name == 'pima_diabetes':
            df = self.load_pima_diabetes()
            target_col = 'Outcome'
            # Exclude target and derived columns
            feature_cols = [col for col in df.columns 
                          if col not in ['Outcome', 'AgeGroup']]
        elif dataset_name == 'adult_income':
            df = self.load_adult_income()
            target_col = 'income'
            # Adult dataset sensitive column is 'age'
            feature_cols = [col for col in df.columns if col != 'income']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Remove missing values
        df = df.dropna()
        
        # Separate features and target
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_sensitive_attributes(self, dataset_name: str) -> list:
        """
        Get sensitive attributes for a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            List of sensitive attribute names
        """
        datasets_config = self.config.get('data.datasets', [])
        
        for dataset in datasets_config:
            if dataset['name'] == dataset_name:
                return dataset['sensitive_attributes']
        
        return []
