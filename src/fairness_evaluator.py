"""Fairness evaluation utilities using Fairlearn and AIF360."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

# Fairlearn imports
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    true_positive_rate_difference,
    false_positive_rate_difference,
    MetricFrame
)

# AIF360 imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("Warning: AIF360 not available. Some fairness metrics will be unavailable.")


class FairnessEvaluator:
    """Evaluate fairness of machine learning models."""
    
    def __init__(self):
        """Initialize fairness evaluator."""
        self.fairness_results = {}
    
    def evaluate_fairness_fairlearn(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_features: pd.DataFrame,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate fairness using Fairlearn metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attributes (e.g., age, gender, race)
            model_name: Name of the model
            
        Returns:
            Dictionary of fairness metrics
        """
        fairness_metrics = {}
        
        # Calculate demographic parity
        try:
            dp_diff = demographic_parity_difference(
                y_true, y_pred,
                sensitive_features=sensitive_features
            )
            dp_ratio = demographic_parity_ratio(
                y_true, y_pred,
                sensitive_features=sensitive_features
            )
            fairness_metrics['demographic_parity_difference'] = dp_diff
            fairness_metrics['demographic_parity_ratio'] = dp_ratio
        except Exception as e:
            print(f"Error calculating demographic parity: {e}")
        
        # Calculate equalized odds
        try:
            eo_diff = equalized_odds_difference(
                y_true, y_pred,
                sensitive_features=sensitive_features
            )
            eo_ratio = equalized_odds_ratio(
                y_true, y_pred,
                sensitive_features=sensitive_features
            )
            fairness_metrics['equalized_odds_difference'] = eo_diff
            fairness_metrics['equalized_odds_ratio'] = eo_ratio
        except Exception as e:
            print(f"Error calculating equalized odds: {e}")
            
        # Add TPR and FPR differences
        try:
            tpr_diff = true_positive_rate_difference(
                y_true, y_pred,
                sensitive_features=sensitive_features
            )
            fpr_diff = false_positive_rate_difference(
                y_true, y_pred,
                sensitive_features=sensitive_features
            )
            fairness_metrics['tpr_difference'] = tpr_diff
            fairness_metrics['fpr_difference'] = fpr_diff
        except Exception as e:
            print(f"Error calculating TPR/FPR differences: {e}")
        
        # Store results
        self.fairness_results[model_name] = fairness_metrics
        
        return fairness_metrics
    
    def evaluate_fairness_aif360(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_attribute: str,
        privileged_groups: List[Dict],
        unprivileged_groups: List[Dict],
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate fairness using AIF360 metrics.
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attribute: Name of sensitive attribute
            privileged_groups: List of privileged group definitions
            unprivileged_groups: List of unprivileged group definitions
            model_name: Name of the model
            
        Returns:
            Dictionary of fairness metrics
        """
        if not AIF360_AVAILABLE:
            print("AIF360 not available. Skipping AIF360 metrics.")
            return {}
        
        fairness_metrics = {}
        
        try:
            # Prepare data for AIF360
            df = X.copy()
            df['label'] = y_true.values
            df['pred'] = y_pred
            
            # Create BinaryLabelDataset for true labels
            dataset_true = BinaryLabelDataset(
                df=df,
                label_names=['label'],
                protected_attribute_names=[sensitive_attribute]
            )
            
            # Create BinaryLabelDataset for predictions
            dataset_pred = dataset_true.copy()
            dataset_pred.labels = y_pred.values.reshape(-1, 1)
            
            # Calculate dataset metrics
            dataset_metric = BinaryLabelDatasetMetric(
                dataset_true,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )
            
            # Calculate classification metrics
            classification_metric = ClassificationMetric(
                dataset_true,
                dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )
            
            # Disparate Impact
            fairness_metrics['disparate_impact'] = dataset_metric.disparate_impact()
            
            # Statistical Parity Difference
            fairness_metrics['statistical_parity_difference'] = \
                dataset_metric.statistical_parity_difference()
            
            # Equal Opportunity Difference
            fairness_metrics['equal_opportunity_difference'] = \
                classification_metric.equal_opportunity_difference()
            
            # Average Odds Difference
            fairness_metrics['average_odds_difference'] = \
                classification_metric.average_odds_difference()
            
            # Theil Index (measures inequality)
            fairness_metrics['theil_index'] = classification_metric.theil_index()
            
        except Exception as e:
            print(f"Error calculating AIF360 metrics: {e}")
        
        # Merge with existing results
        if model_name in self.fairness_results:
            self.fairness_results[model_name].update(fairness_metrics)
        else:
            self.fairness_results[model_name] = fairness_metrics
        
        return fairness_metrics
    
    def evaluate_subgroup_performance(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_features: pd.DataFrame,
        metric_func: callable,
        model_name: str = "model"
    ) -> pd.DataFrame:
        """
        Evaluate performance across different subgroups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attributes
            metric_func: Metric function to evaluate
            model_name: Name of the model
            
        Returns:
            DataFrame with subgroup performance
        """
        # Create MetricFrame
        metric_frame = MetricFrame(
            metrics=metric_func,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        # Get results by group
        results_by_group = metric_frame.by_group
        
        return results_by_group
    
    def compare_fairness_across_models(
        self,
        metric: str = 'demographic_parity_difference'
    ) -> pd.DataFrame:
        """
        Compare fairness metrics across models.
        
        Args:
            metric: Fairness metric to compare
            
        Returns:
            DataFrame with fairness comparison
        """
        if not self.fairness_results:
            print("No fairness results available")
            return pd.DataFrame()
        
        comparison = {}
        
        for model_name, metrics in self.fairness_results.items():
            if metric in metrics:
                comparison[model_name] = metrics[metric]
        
        return pd.DataFrame([comparison]).T.rename(columns={0: metric})
    
    def plot_fairness_comparison(
        self,
        metric: str = 'demographic_parity_difference',
        save_path: Optional[str] = None
    ):
        """
        Plot fairness metric comparison across models.
        
        Args:
            metric: Fairness metric to plot
            save_path: Path to save the plot
        """
        comparison_df = self.compare_fairness_across_models(metric)
        
        if comparison_df.empty:
            print(f"No data available for metric: {metric}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(comparison_df.index, comparison_df[metric])
        
        # Color bars based on fairness threshold
        threshold = 0.1
        colors = ['red' if abs(val) > threshold else 'green' 
                 for val in comparison_df[metric]]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add threshold lines
        plt.axhline(y=threshold, color='orange', linestyle='--', 
                   label=f'Fairness threshold (+{threshold})')
        plt.axhline(y=-threshold, color='orange', linestyle='--', 
                   label=f'Fairness threshold (-{threshold})')
        
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Fairness Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_subgroup_performance(
        self,
        subgroup_results: pd.DataFrame,
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None
    ):
        """
        Plot performance across subgroups.
        
        Args:
            subgroup_results: DataFrame with subgroup performance
            metric_name: Name of the metric
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(subgroup_results, pd.Series):
            subgroup_results = subgroup_results.to_frame(name=metric_name)
        
        # Create bar plot
        ax = subgroup_results.plot(kind='bar', legend=False)
        
        plt.xlabel('Subgroup')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} by Subgroup')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_fairness_report(
        self,
        model_name: str
    ) -> str:
        """
        Generate a comprehensive fairness report.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted fairness report string
        """
        if model_name not in self.fairness_results:
            return f"No fairness results available for {model_name}"
        
        metrics = self.fairness_results[model_name]
        
        report = f"\n{'='*60}\n"
        report += f"Fairness Report - {model_name}\n"
        report += f"{'='*60}\n\n"
        
        for metric_name, value in metrics.items():
            report += f"{metric_name.replace('_', ' ').title()}: {value:.4f}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
