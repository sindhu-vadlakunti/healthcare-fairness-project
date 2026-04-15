"""SHAP analysis utilities for model explainability."""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """Analyze model predictions using SHAP values."""
    
    def __init__(self):
        """Initialize SHAP analyzer."""
        self.explainers = {}
        self.shap_values = {}
    
    def create_explainer(
        self,
        model: Any,
        X_train: pd.DataFrame,
        model_name: str = "model",
        explainer_type: str = "auto"
    ):
        """
        Create SHAP explainer for a model.
        
        Args:
            model: Trained model
            X_train: Training data for background
            model_name: Name of the model
            explainer_type: Type of explainer ('tree', 'linear', 'kernel', 'auto')
        """
        print(f"Creating SHAP explainer for {model_name}...")
        
        if explainer_type == "auto":
            # Auto-detect explainer type
            try:
                # Handle LightGBM explicitly (often causes name-setter issues)
                if 'LGBM' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                elif 'XGB' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                elif hasattr(model, 'tree_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer as fallback (slower but works for any model)
                    print(f"  [Action] Sampling 50 rows for background data (KernelExplainer)...")
                    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
            except Exception as e:
                print(f"  [Warning] SHAP Explainer detection failed ({e}). Falling back to KernelExplainer.")
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
        
        elif explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
        
        elif explainer_type == "linear":
            explainer = shap.LinearExplainer(model, X_train)
        
        elif explainer_type == "kernel":
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
        
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        self.explainers[model_name] = explainer
        
        return explainer
    
    def calculate_shap_values(
        self,
        model_name: str,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate SHAP values for test data.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            
        Returns:
            SHAP values array
        """
        if model_name not in self.explainers:
            raise ValueError(f"Explainer not created for {model_name}")
        
        print(f"Calculating SHAP values for {model_name}...")
        
        explainer = self.explainers[model_name]
        
        # PERFORMANCE OPTIMIZATION: Sample X_test if it's too large (especially for Adult dataset)
        X_test_sample = X_test
        if len(X_test) > 200:
            print(f"  [Action] Sampling 200 rows for SHAP evaluation to save time...")
            X_test_sample = shap.sample(X_test, 200)

        shap_values = explainer.shap_values(X_test_sample)
        
        # For binary classification, some explainers return values for both classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        self.shap_values[model_name] = shap_values
        
        return shap_values
    
    def plot_summary(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        plot_type: str = "dot",
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP summary plot.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance based on mean absolute SHAP values.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(max_display)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return feature_importance
    
    def plot_dependence(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            feature: Feature to plot
            interaction_feature: Feature to use for coloring (auto-detected if None)
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            X_test,
            interaction_index=interaction_feature,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_force(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        instance_index: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            instance_index: Index of instance to explain
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        # Create force plot
        shap.force_plot(
            explainer.expected_value,
            shap_values[instance_index],
            X_test.iloc[instance_index],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        instance_index: int = 0,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            instance_index: Index of instance to explain
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_index],
            base_values=explainer.expected_value,
            data=X_test.iloc[instance_index].values,
            feature_names=X_test.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_fairness_with_shap(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        sensitive_feature: str
    ) -> pd.DataFrame:
        """
        Analyze how sensitive features contribute to predictions.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            sensitive_feature: Name of sensitive feature
            
        Returns:
            DataFrame with SHAP value statistics for sensitive feature
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        
        # Get index of sensitive feature
        feature_idx = X_test.columns.get_loc(sensitive_feature)
        
        # Get SHAP values for sensitive feature
        sensitive_shap = shap_values[:, feature_idx]
        
        # Calculate statistics
        stats = pd.DataFrame({
            'mean_shap': [sensitive_shap.mean()],
            'abs_mean_shap': [np.abs(sensitive_shap).mean()],
            'std_shap': [sensitive_shap.std()],
            'min_shap': [sensitive_shap.min()],
            'max_shap': [sensitive_shap.max()]
        })
        
        print(f"\nSHAP Analysis for {sensitive_feature}:")
        print(stats)
        
        return stats
    
    def get_feature_importance_ranking(
        self,
        model_name: str,
        X_test: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get feature importance ranking based on SHAP values.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            
        Returns:
            DataFrame with feature importance ranking
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(model_name, X_test)
        
        shap_values = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        importance['rank'] = range(1, len(importance) + 1)
        
        return importance
