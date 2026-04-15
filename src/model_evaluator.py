"""Model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate machine learning models."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def evaluate_multiple_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with evaluation metrics for all models
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        
        return results_df
    
    def get_confusion_matrix(
        self,
        model_name: str
    ) -> np.ndarray:
        """
        Get confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Confusion matrix
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        return confusion_matrix(results['y_true'], results['y_pred'])
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = self.get_confusion_matrix(model_name)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curve(
        self,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        
        if results['y_pred_proba'] is None:
            print(f"Model {model_name} does not support probability predictions")
            return
        
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
        roc_auc = roc_auc_score(results['y_true'], results['y_pred_proba'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot precision-recall curve.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        
        if results['y_pred_proba'] is None:
            print(f"Model {model_name} does not support probability predictions")
            return
        
        precision, recall, _ = precision_recall_curve(
            results['y_true'],
            results['y_pred_proba']
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def compare_models(
        self,
        metric: str = 'roc_auc',
        save_path: Optional[str] = None
    ):
        """
        Compare models using a specific metric.
        
        Args:
            metric: Metric to compare
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            print("No models evaluated yet")
            return
        
        model_names = []
        metric_values = []
        
        for model_name, results in self.evaluation_results.items():
            if metric in results['metrics']:
                model_names.append(model_name)
                metric_values.append(results['metrics'][metric])
        
        if not model_names:
            print(f"Metric {metric} not available for any model")
            return
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def get_classification_report(
        self,
        model_name: str
    ) -> str:
        """
        Get classification report for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Classification report string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        return classification_report(results['y_true'], results['y_pred'])
