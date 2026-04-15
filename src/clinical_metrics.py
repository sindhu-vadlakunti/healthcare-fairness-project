"""Clinical-specific evaluation metrics for healthcare publication."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from typing import Any, List


class ClinicalEvaluator:
    """Specialized metrics for clinical model validation."""
    
    def plot_calibration_by_group(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: pd.Series,
        n_bins: int = 10,
        save_path: str = None
    ):
        """
        Plot calibration curves for different subgroups.
        """
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        # Consistent with other methods: use sensitive_features Series
        groups = sensitive_features.unique()
        
        for group in groups:
            mask = sensitive_features == group
            if mask.sum() == 0: continue
            
            # Use X_test for prediction and y_test for truth, both masked by sensitive_features
            y_proba = model.predict_proba(X_test[mask])[:, 1]
            prob_true, prob_pred = calibration_curve(y_test[mask], y_proba, n_bins=n_bins)
            
            plt.plot(prob_pred, prob_true, "s-", label=f"Group: {group}")
            
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title(f"Calibration Curves by Subgroup")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Formula: ECE = sum( (bin_count/total_count) * abs(acc(bin) - conf(bin)) )
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated intermediate values
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

    def subgroup_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_features: pd.Series,
        save_path: str = None
    ):
        """
        Create a grid of confusion matrices for each subgroup.
        """
        groups = sensitive_features.unique()
        n_groups = len(groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
        if n_groups == 1: axes = [axes]
        
        for i, group in enumerate(groups):
            mask = sensitive_features == group
            cm = confusion_matrix(y_true[mask], y_pred[mask], normalize='true')
            
            sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(f"Subgroup: {group}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def risk_stratified_analysis(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        sensitive_features: pd.Series,
        thresholds: List[float] = [0.0, 0.3, 0.7, 1.0],
        labels: List[str] = ['Low Risk', 'Medium Risk', 'High Risk']
    ) -> pd.DataFrame:
        """
        Analyze fairness and accuracy within specific risk tiers.
        
        Proof Value: Proves if bias is concentrated in the critical 'High Risk' group.
        """
        risk_results = []
        
        for i in range(len(thresholds) - 1):
            lower = thresholds[i]
            upper = thresholds[i+1]
            tier_name = labels[i]
            
            mask = (y_proba >= lower) & (y_proba < upper)
            if mask.sum() == 0: continue
            
            # Use iloc for numpy-based masking
            tier_y_true = y_true.iloc[mask] if isinstance(y_true, (pd.Series, pd.DataFrame)) else y_true[mask]
            tier_y_proba = y_proba[mask]
            tier_sensitive = sensitive_features.iloc[mask] if isinstance(sensitive_features, (pd.Series, pd.DataFrame)) else sensitive_features[mask]
            
            # Simple accuracy for the tier: use the threshold center 0.5 as decision boundary
            tier_y_pred = (tier_y_proba >= 0.5).astype(int)
            tier_acc = accuracy_score(tier_y_true, tier_y_pred) if len(tier_y_true) > 0 else 0
            
            # Subgroup metrics in this tier
            groups = tier_sensitive.unique()
            group_accs = {}
            for group in groups:
                g_mask = tier_sensitive == group
                if g_mask.sum() > 0:
                    g_y_true = tier_y_true.iloc[g_mask.values] if isinstance(tier_y_true, pd.Series) else tier_y_true[g_mask]
                    g_y_pred = tier_y_pred[g_mask.values] if isinstance(g_mask, pd.Series) else tier_y_pred[g_mask]
                    group_accs[f'acc_{group}'] = accuracy_score(g_y_true, g_y_pred)
                else:
                    group_accs[f'acc_{group}'] = 0
            
            res = {
                'tier': tier_name,
                'count': int(mask.sum()),
                'overall_acc': tier_acc,
                **group_accs
            }
            risk_results.append(res)
            
        return pd.DataFrame(risk_results)
    def get_subgroup_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_features: pd.Series,
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Calculate detailed metrics per subgroup (TPR, FPR, etc.).
        """
        from sklearn.metrics import recall_score, precision_score, f1_score
        
        groups = sensitive_features.unique()
        rows = []
        
        for group in groups:
            mask = sensitive_features == group
            gt = y_true[mask]
            pred = y_pred[mask]
            
            tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
            
            rows.append({
                'Model': model_name,
                'Group': str(group),
                'TPR (Recall)': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'Accuracy': accuracy_score(gt, pred),
                'Precision': precision_score(gt, pred, zero_division=0),
                'F1-Score': f1_score(gt, pred, zero_division=0)
            })
            
        df = pd.DataFrame(rows)
        return df
