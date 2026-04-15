"""Advanced research analysis tools for publication-quality results."""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score


class ResearchAnalyzer:
    """Advanced tools for statistical rigor and research analysis."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def bootstrap_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_iterations: int = 1000,
        metrics: List[str] = ['accuracy', 'roc_auc']
    ) -> pd.DataFrame:
        """
        Calculate metrics with 95% Confidence Intervals using bootstrapping.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_iterations: Number of bootstrap samples
            metrics: Metrics to calculate
            
        Returns:
            DataFrame with mean, low (2.5%), and high (97.5%) values
        """
        print(f"Bootstrapping {n_iterations} iterations...")
        bootstrapped_stats = []
        
        # Temporarily disable parallelism for the model to prevent MemoryError on Windows
        original_n_jobs = getattr(model, 'n_jobs', None)
        if hasattr(model, 'n_jobs'):
            try:
                model.n_jobs = 1
            except Exception:
                pass
                
        try:
            # Predict ONCE outside the loop to save time
            y_pred_all = model.predict(X_test)
            y_proba_all = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            for i in range(n_iterations):
                # Sample indices instead of raw data to keep y_pred matched
                indices = resample(
                    range(len(y_test)), 
                    random_state=self.random_seed + i
                )
                
                y_resample = y_test.iloc[indices]
                y_pred_resample = y_pred_all[indices]
                
                iter_results = {}
                if 'accuracy' in metrics:
                    iter_results['accuracy'] = accuracy_score(y_resample, y_pred_resample)
                if 'roc_auc' in metrics and y_proba_all is not None:
                    y_proba_resample = y_proba_all[indices]
                    iter_results['roc_auc'] = roc_auc_score(y_resample, y_proba_resample)
                    
                bootstrapped_stats.append(iter_results)
        finally:
            # Restore original parallelism
            if original_n_jobs is not None:
                try:
                    model.n_jobs = original_n_jobs
                except Exception:
                    pass
            
        df = pd.DataFrame(bootstrapped_stats)
        
        results = pd.DataFrame({
            'mean': df.mean(),
            'ci_low': df.quantile(0.025),
            'ci_high': df.quantile(0.975),
            'std': df.std()
        })
        
        return results

    def bootstrap_fairness_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: pd.Series,
        n_iterations: int = 100,
        metrics: List[str] = ['dp_diff', 'eo_diff'],
        return_samples: bool = False
    ) -> Any:
        """
        Calculate Fairness Metrics with 95% Confidence Intervals.
        
        Proof Value: Proves that fairness results are statistically stable.
        """
        from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
        
        print(f"Bootstrapping fairness metrics ({n_iterations} iterations)...")
        bootstrapped_fairness = []
        
        # Reset indices for consistent sampling
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        sensitive_reset = sensitive_features.reset_index(drop=True)
        
        # Temporarily disable parallelism for the model to prevent MemoryError on Windows
        original_n_jobs = getattr(model, 'n_jobs', None)
        if hasattr(model, 'n_jobs'):
            try:
                model.n_jobs = 1
            except Exception:
                pass

        try:
            # Predict ONCE outside the loop
            y_pred_all = model.predict(X_test_reset)
            
            for i in range(n_iterations):
                indices = resample(
                    range(len(y_test_reset)), 
                    random_state=self.random_seed + i
                )
                
                y_true_s = y_test_reset.iloc[indices]
                sensitive_s = sensitive_reset.iloc[indices]
                y_pred_s = y_pred_all[indices]
                
                iter_res = {}
                if 'dp_diff' in metrics:
                    iter_res['dp_diff'] = demographic_parity_difference(y_true_s, y_pred_s, sensitive_features=sensitive_s)
                if 'eo_diff' in metrics:
                    iter_res['eo_diff'] = equalized_odds_difference(y_true_s, y_pred_s, sensitive_features=sensitive_s)
                    
                bootstrapped_fairness.append(iter_res)
        finally:
            # Restore original parallelism
            if original_n_jobs is not None:
                try:
                    model.n_jobs = original_n_jobs
                except Exception:
                    pass
            
        df = pd.DataFrame(bootstrapped_fairness)
        if return_samples:
            return df
            
        results = pd.DataFrame({
            'mean': df.mean(),
            'ci_low': df.quantile(0.025),
            'ci_high': df.quantile(0.975),
            'std': df.std()
        })
        
        return results

    def calculate_pareto_frontier(
        self,
        results_df: pd.DataFrame,
        perf_metric: str,
        fair_metric: str,
        higher_perf_is_better: bool = True,
        lower_fair_is_better: bool = True
    ) -> pd.DataFrame:
        """
        Identify models on the Pareto Frontier (optimal trade-offs).
        
        Args:
            results_df: DataFrame with all model results
            perf_metric: Column for performance
            fair_metric: Column for fairness violation (e.g., DP Difference)
            
        Returns:
            Subset of results_df containing only Pareto-optimal models
        """
        # Prepare data (handle absolute values for fairness)
        data = results_df.copy()
        if 'difference' in fair_metric.lower() or 'impact' in fair_metric.lower():
            data[f'abs_{fair_metric}'] = data[fair_metric].abs()
            fair_col = f'abs_{fair_metric}'
        else:
            fair_col = fair_metric

        pareto_indices = []
        for i, row_i in data.iterrows():
            is_dominated = False
            for j, row_j in data.iterrows():
                if i == j: continue
                
                # J dominates I if it's better in one and at least equal in others
                better_perf = row_j[perf_metric] > row_i[perf_metric] if higher_perf_is_better else row_j[perf_metric] < row_i[perf_metric]
                better_fair = row_j[fair_col] < row_i[fair_col] if lower_fair_is_better else row_j[fair_col] > row_i[fair_col]
                
                equal_perf = row_j[perf_metric] == row_i[perf_metric]
                equal_fair = row_j[fair_col] == row_i[fair_col]
                
                if (better_perf and (better_fair or equal_fair)) or (better_fair and (better_perf or equal_perf)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
                
        return data.loc[pareto_indices].sort_values(perf_metric, ascending=not higher_perf_is_better)

    def statistical_significance_test(
        self,
        baseline_scores: np.ndarray,
        modified_scores: np.ndarray,
        test_type: str = 't-test',
        higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical testing suite for model comparison.
        """
        results = {}
        
        # 1. Selection of test
        if test_type == 'wilcoxon':
            stat, p_val = stats.wilcoxon(baseline_scores, modified_scores)
            results['test_name'] = 'Wilcoxon signed-rank'
        else:
            # Paired t-test is more appropriate for before/after comparison
            stat, p_val = stats.ttest_rel(baseline_scores, modified_scores)
            results['test_name'] = 'Paired t-test'
            
        results['statistic'] = stat
        results['p_value'] = p_val
        
        # 2. Cohen's d effect size
        d = self.calculate_cohens_d(baseline_scores, modified_scores)
        results['cohens_d'] = d
        
        # 3. Bootstrap CI for the difference
        diffs = modified_scores - baseline_scores
        boot_diffs = []
        for i in range(1000):
            sample = resample(diffs, random_state=self.random_seed + i)
            boot_diffs.append(np.mean(sample))
        
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        results['ci_diff'] = (ci_low, ci_high)
        
        results['significant'] = p_val < 0.05
        
        mean_diff = np.mean(diffs)
        if higher_is_better:
            results['interpretation'] = "Significant Improvement" if (p_val < 0.05 and mean_diff > 0) else \
                                        "Significant Degradation" if (p_val < 0.05 and mean_diff < 0) else \
                                        "Not Significant"
        else:
            # For fairness (DP Diff), lower is better
            results['interpretation'] = "Significant Improvement" if (p_val < 0.05 and mean_diff < 0) else \
                                        "Significant Amplification" if (p_val < 0.05 and mean_diff > 0) else \
                                        "Not Significant"
        
        return results

    def calculate_cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(x), len(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0: return 0
        return (np.mean(x) - np.mean(y)) / pooled_std

    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform McNemar's test for paired nominal data.
        Used to determine if there is a statistically significant difference 
        between the error rates of two models.
        """
        # Contingency table
        #           Model B correct | Model B wrong
        # Model A corr |   a           |      b
        # Model A wro  |   c           |      d
        
        a = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
        b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
        c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        d = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))
        
        # McNemar's test statistic
        # (abs(b - c) - 1)^2 / (b + c)
        if (b + c) > 0:
            stat = (np.abs(b - c) - 1)**2 / (b + c)
            p_val = 1 - stats.chi2.cdf(stat, 1)
        else:
            stat, p_val = 0, 1.0
            
        return {
            'statistic': stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'contingency_table': [[a, b], [c, d]]
        }
