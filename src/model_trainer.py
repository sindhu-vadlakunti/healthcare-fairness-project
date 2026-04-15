"""Model training utilities."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from typing import Dict, Any, Tuple, Optional
import joblib
from pathlib import Path


class ModelTrainer:
    """Train and evaluate machine learning models."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
    
    def get_baseline_models(self) -> Dict[str, Any]:
        """
        Get baseline models.
        
        Returns:
            Dictionary of baseline model instances
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000
            ),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_seed,
                max_depth=10
            )
        }
        return models
    
    def get_ensemble_models(self) -> Dict[str, Any]:
        """
        Get ensemble models.
        
        Returns:
            Dictionary of ensemble model instances
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_seed,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                random_state=self.random_seed,
                n_jobs=-1,
                verbose=-1
            )
        }
        return models
    
    def get_voting_ensemble(
        self,
        estimators: Optional[list] = None
    ) -> VotingClassifier:
        """
        Create voting ensemble.
        
        Args:
            estimators: List of (name, estimator) tuples
            
        Returns:
            VotingClassifier instance
        """
        if estimators is None:
            ensemble_models = self.get_ensemble_models()
            estimators = [(name, model) for name, model in ensemble_models.items()]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'baseline'
    ) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            model_type: Type of model ('baseline' or 'ensemble')
            
        Returns:
            Trained model
        """
        # Get model
        if model_type == 'baseline':
            models = self.get_baseline_models()
        else:
            models = self.get_ensemble_models()
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = models[model_name]
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        include_baseline: bool = True,
        include_ensemble: bool = True,
        include_voting: bool = True
    ) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            include_baseline: Whether to train baseline models
            include_ensemble: Whether to train ensemble models
            include_voting: Whether to train voting ensemble
            
        Returns:
            Dictionary of trained models
        """
        trained = {}
        
        # Train baseline models
        if include_baseline:
            baseline_models = self.get_baseline_models()
            for name, model in baseline_models.items():
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                trained[name] = model
        
        # Train ensemble models
        if include_ensemble:
            ensemble_models = self.get_ensemble_models()
            for name, model in ensemble_models.items():
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                trained[name] = model
        
        # Train voting ensemble
        if include_voting and include_ensemble:
            print("Training voting_ensemble...")
            voting_model = self.get_voting_ensemble()
            voting_model.fit(X_train, y_train)
            trained['voting_ensemble'] = voting_model
        
        self.trained_models = trained
        return trained
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, list],
        cv: int = 5
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        # Get base model
        all_models = {**self.get_baseline_models(), **self.get_ensemble_models()}
        
        if model_name not in all_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        base_model = all_models[model_name]
        
        # Perform grid search
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params[model_name] = grid_search.best_params_
        self.trained_models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std of scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def save_model(self, model_name: str, save_path: Path):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_models[model_name], save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_name: str, load_path: Path):
        """
        Load trained model from disk.
        
        Args:
            model_name: Name to assign to the model
            load_path: Path to load the model from
        """
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        self.trained_models[model_name] = joblib.load(load_path)
        print(f"Model loaded from {load_path}")
