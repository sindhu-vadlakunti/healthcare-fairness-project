"""Visualizer for publication-ready figures."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class PaperVisualizer:
    """Style settings and plot types specifically for research papers."""
    
    def __init__(self):
        # Set academic style with fallbacks
        sns.set_context("paper", font_scale=1.5)
        try:
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
            plt.rcParams['font.family'] = 'serif'
        except Exception:
            # Fallback to default if serif family setting fails
            pass
        plt.rcParams['axes.labelweight'] = 'bold'
        
    def plot_pareto_frontier(
        self,
        results_df: pd.DataFrame,
        pareto_df: pd.DataFrame,
        perf_col: str,
        fair_col: str,
        save_path: str = None
    ):
        """
        Custom Pareto scatter plot with frontier line.
        """
        plt.figure(figsize=(10, 7))
        
        # Plot all models
        plt.scatter(
            results_df[fair_col].abs(), 
            results_df[perf_col], 
            c='lightgrey', alpha=0.5, label='Other Configurations'
        )
        
        # Plot Pareto frontier
        plt.plot(
            pareto_df[fair_col].abs(), 
            pareto_df[perf_col], 
            'r--', alpha=0.8
        )
        plt.scatter(
            pareto_df[fair_col].abs(), 
            pareto_df[perf_col], 
            c='red', s=100, label='Pareto Optimal'
        )
        
        # Annotate Pareto models
        for i, row in pareto_df.iterrows():
            plt.annotate(i, (abs(row[fair_col]), row[perf_col]), xytext=(5, 5), textcoords='offset points')
            
        plt.xlabel(f"{fair_col.replace('_', ' ').title()} (Violation)")
        plt.ylabel(perf_col.replace('_', ' ').title())
        plt.title("Fairness-Performance Pareto Frontier")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """Graph 1: Overlay ROC curves for all models."""
        from sklearn.metrics import roc_curve, auc
        plt.figure(figsize=(10, 8))
        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_fairness_bar_chart(self, fair_df: pd.DataFrame, metric: str = 'Demographic Parity Diff', save_path: str = None):
        """Graph 3: Demographic Parity Bar Chart with threshold line."""
        plt.figure(figsize=(12, 6))
        colors = ['green' if x < 0.1 else 'red' for x in fair_df[metric]]
        sns.barplot(x='Model', y=metric, data=fair_df, palette=colors)
        plt.axhline(0.1, color='red', linestyle='--', label='Threshold (0.1)')
        plt.xticks(rotation=45)
        plt.title(f'Fairness Comparison: {metric}')
        plt.legend()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_subgroup_heatmap(self, sg_df: pd.DataFrame, metric: str = 'Accuracy', save_path: str = None):
        """Graph 4: Subgroup Performance Heatmap."""
        pivot = sg_df.pivot(index="Model", columns="Group", values=metric)
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn")
        plt.title(f'Subgroup Performance Heatmap ({metric})')
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_significance_heatmap(self, sig_df: pd.DataFrame, save_path: str = None):
        """Graph 10: Statistical Significance Heatmap."""
        if sig_df.empty: return
        # Convert p-values to numeric if stored as strings
        sig_df['p_val_num'] = pd.to_numeric(sig_df['p_value'], errors='coerce').fillna(0)
        pivot = sig_df.pivot(index="Comparison", columns="Metric", values="p_val_num")
        plt.figure(figsize=(10, 6))
        # Mask for significance colors
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", center=0.05)
        plt.title('Statistical Significance (p-values)')
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cv_boxplots(self, cv_results: List[np.ndarray], labels: List[str], save_path: str = None):
        """Graph 11: Cross-Validation Box Plots."""
        plt.figure(figsize=(12, 6))
        plt.boxplot(cv_results, labels=labels)
        plt.ylabel("Accuracy / AUC")
        plt.title("Model Stability (Cross-Validation)")
        plt.xticks(rotation=45)
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mitigation_impact(self, mit_df: pd.DataFrame, save_path: str = None):
        """Graph 9: Bias Mitigation Impact (Before/After)."""
        if mit_df.empty: return
        plt.figure(figsize=(12, 6))
        # Melt for seaborn
        melted = mit_df.melt(id_vars=['Model', 'Strategy'], value_vars=['DP Diff (Before)', 'DP Diff (After)'], 
                             var_name='Stage', value_name='Bias')
        sns.barplot(x='Model', y='Bias', hue='Stage', data=melted)
        plt.title('Impact of Bias Mitigation')
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_fairness_confusion_matrices(self, y_true, y_pred, sensitive_features, save_path=None):
        """Graph 13: Side-by-side Confusion Matrices for each demographic group."""
        from sklearn.metrics import confusion_matrix
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
        if n_groups == 1: axes = [axes]
        
        for i, group in enumerate(groups):
            mask = (sensitive_features == group)
            cm = confusion_matrix(y_true[mask], y_pred[mask], normalize='true')
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=axes[i], cbar=False)
            axes[i].set_title(f"Group: {group}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
            
        plt.suptitle("Confusion Matrices by Subgroup", fontsize=20, y=1.05)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_complexity_vs_fairness(self, comp_df, save_path=None):
        """Graph 14: Complexity vs Fairness Trade-off."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Accuracy', y='DP Difference', marker='o', data=comp_df, sort=False)
        for i, row in comp_df.iterrows():
            plt.annotate(row['Complexity'], (row['Accuracy'], row['DP Difference']), 
                         xytext=(10, -5), textcoords='offset points')
            
        plt.title("Model Complexity vs Fairness Trade-off")
        plt.xlabel("Accuracy (Performance)")
        plt.ylabel("Demographic Parity Difference (Bias)")
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_calibration_curve_subgroups(self, model, X_test, y_test, sensitive_features, save_path=None):
        """Graph 12: Calibration curve stratified by demographic groups."""
        from sklearn.calibration import calibration_curve
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        groups = np.unique(sensitive_features)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        if y_proba is not None:
            for group in groups:
                mask = (sensitive_features == group)
                if mask.any():
                    prob_true, prob_pred = calibration_curve(y_test[mask], y_proba[mask], n_bins=5)
                    plt.plot(prob_pred, prob_true, "s-", label=f"Group: {group}")
        
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title("Calibration Curves by Subgroup")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
