"""Unified Research Engine for Healthcare Fairness Publication."""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample

# Internal Imports
from src.config_loader import get_config
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.fairness_evaluator import FairnessEvaluator
from src.research_analysis import ResearchAnalyzer
from src.clinical_metrics import ClinicalEvaluator
from src.paper_visualizer import PaperVisualizer
from src.automated_reporting import PaperReporter
from src.shap_analyzer import SHAPAnalyzer
from src.bias_mitigation import BiasMitigator
from src.data_ethics import DataEthicsProfiler

class ResearchEngine:
    """A professional-grade engine to orchestrate all healthcare fairness research."""
    
    def __init__(self, output_root: str = "results/manuscript_assets", interactive: bool = False):
        self.config = get_config()
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        
        if not self.interactive:
            import matplotlib
            matplotlib.use('Agg')
            print("  [Note] Headless mode: Figures will be saved directly to disk.")
        
        # Initialize Core Modules
        self.loader = DataLoader()
        self.trainer = ModelTrainer(random_seed=42)
        self.evaluator = ModelEvaluator()
        self.fairness_eval = FairnessEvaluator()
        
        # Initialize Research Modules
        self.research_analyser = ResearchAnalyzer()
        self.clinical_eval = ClinicalEvaluator()
        self.paper_vis = PaperVisualizer()
        self.reporter = PaperReporter()
        self.shap_analyzer = SHAPAnalyzer()
        self.mitigator = BiasMitigator()
        self.ethics_profiler = DataEthicsProfiler()
        
        self.all_dataset_results = {}

    def get_sensitive_bins(self, X: pd.DataFrame, dataset_name: str) -> pd.Series:
        """Centralized binning for all fairness analysis matching professional standards."""
        if dataset_name == 'pima_diabetes':
            # BUG FIX: Merged >60 into 40-60 to avoid small sample size (n=5) instability
            return pd.cut(X['Age'], bins=[0, 40, 100], labels=['<40', '>=40'])
        elif dataset_name == 'adult_income':
            # Adult dataset uses lower-case 'age'
            return pd.cut(X['age'], bins=[0, 40, 60, 100], labels=['<40', '40-60', '>60'])
        return None

    def run_ablation_study(self, dataset_name: str = 'pima_diabetes') -> pd.DataFrame:
        """
        Deep ablation study with 7 variants.
        BUG FIX: Each variant now uses a stateful FeatureEngineer that fits on
        X_train then transforms X_test using only train-derived parameters.
        The old zero-fill approach caused data leakage and invalid fairness metrics.
        """
        print(f"  -> Deep Ablation: Orchestrating 7 variants...")
        X_train, X_test, y_train, y_test = self.loader.prepare_dataset(dataset_name)
        age_bins_test = self.get_sensitive_bins(X_test, dataset_name)

        variants = [
            # (Name, Interactions, Polynomial, Binning, Scaling, Selection)
            ("Baseline",          False, False, False, False, False),
            ("+Interactions Only", True, False, False, False, False),
            ("+Polynomial Only",  False, True,  False, False, False),
            ("+Binning Only",     False, False, True,  False, False),
            ("+Scaling Only",     False, False, False, True,  False),
            ("+Selection Only",   False, False, False, False, True),
            ("Full Pipeline",     True,  True,  True,  True,  True),
        ]

        results = []
        for name, inter, poly, binning, scale, select in variants:
            eng = FeatureEngineer()
            X_tr, X_te = X_train.copy(), X_test.copy()

            # ---- FIT on train, TRANSFORM test (no leakage) ----
            if inter or poly or binning or select:
                X_tr = eng.fit_transform(
                    X_tr, y_train,
                    include_interactions=inter,
                    include_polynomials=poly,
                    include_binning=binning,
                    select_features=select,
                )
                X_te = eng.transform(X_te)   # uses only train-learned parameters

            # Preprocess (scaler fitted on train only)
            prep = DataPreprocessor(scaling_method='standard' if scale else 'minmax').fit(X_tr)
            X_tr_p = prep.transform(X_tr)
            X_te_p = prep.transform(X_te)

            model = self.trainer.train_model('random_forest', X_tr_p, y_train, 'ensemble')
            y_pred = model.predict(X_te_p)

            fairness = self.fairness_eval.evaluate_fairness_fairlearn(
                y_test, y_pred, age_bins_test, f"ab_v_{name}"
            )
            results.append({
                'Variant': name,
                'Accuracy': model.score(X_te_p, y_test),
                'DP Difference': fairness.get('demographic_parity_difference', 0),
                'EO Difference': fairness.get('equalized_odds_difference', 0),
            })

        return pd.DataFrame(results)

    def run_complexity_fairness_analysis(self, X_tr, X_te, y_tr, y_te, age_bins) -> pd.DataFrame:
        """Analyze if model complexity (depth/estimators) affects fairness (Addition #4)."""
        print("  -> Complexity vs Fairness Sweep...")
        from sklearn.ensemble import RandomForestClassifier
        
        complexity_configs = [
            ('Simple', 3, 10),
            ('Medium', 6, 50),
            ('Complex', 12, 100)
        ]
        
        comp_results = []
        for name, depth, est in complexity_configs:
            m = RandomForestClassifier(max_depth=depth, n_estimators=est, random_state=42)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            
            f = self.fairness_eval.evaluate_fairness_fairlearn(y_te, y_pred, age_bins, f"comp_{name}")
            comp_results.append({
                'Complexity': name,
                'Accuracy': m.score(X_te, y_te),
                'DP Difference': f.get('demographic_parity_difference', 0)
            })
            
        return pd.DataFrame(comp_results)

    def generate_manuscript_tables(self):
        """Unified asset generation for all datasets processed in the current run."""
        report_str = self.reporter.generate_manuscript_report(
            self.all_dataset_results, 
            save_path=self.output_root / "PUBLICATION_REPORT.md"
        )
        self.reporter.print_to_terminal(report_str)
        print(f"\n✅ FULL CONSOLIDATED PUBLICATION PACK READY IN: {self.output_root.absolute()}")

    def run_publication_pipeline(self, dataset_names: List[str] = ['pima_diabetes', 'adult_income']):
        """The 'Professional' run that handles multiple datasets and valid demographic binning."""
        self.all_dataset_results = {}
        for ds_name in dataset_names:
            self.all_dataset_results[ds_name] = self._execute_single_dataset_pipeline(ds_name)
        
        # Finally generate assets using all results
        self.generate_manuscript_tables()

    def _execute_single_dataset_pipeline(self, dataset_name: str) -> Dict[str, Any]:
        print(f"\n🏛️ STARTING PROFESSIONAL PUBLICATION RUN: {dataset_name}")
        print("="*60 + "\n")
        
        current_data = {}
        X_train, X_test, y_train, y_test = self.loader.prepare_dataset(dataset_name)
        
        # Get binned features for fairness evaluation
        age_bins_test = self.get_sensitive_bins(X_test, dataset_name)
        age_bins_train = self.get_sensitive_bins(X_train, dataset_name)
        sensitive_col = 'Age' if dataset_name == 'pima_diabetes' else 'age'
        
        # 1. PERFORMANCE & FAIRNESS (Table 1 & 2)
        print("-> Computing Performance and Fairness for 14 variants...")
        perf_rows = []
        fair_rows = []
        cv_rows = []
        cv_fair_rows = []
        
        models_to_test = ['logistic_regression', 'naive_bayes', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm']
        all_models_trained = {}
        
        # Track for best model selection later
        best_auc = 0
        best_id = None
        best_m = None
        X_te_best_p = None
        X_tr_best_p = None
        X_te_base_p = None
        X_tr_base_p = None

        for fe_enabled in [False, True]:
            fe_label = "Yes" if fe_enabled else "No"
            X_tr, X_te = X_train.copy(), X_test.copy()

            if fe_enabled:
                engineer = FeatureEngineer()
                # BUG FIX: Use stateful fit_transform/transform to prevent data leakage.
                # Previously engineer_all_features was called on X_test directly and
                # missing columns were zero-filled, producing corrupt fairness metrics.
                X_tr = engineer.fit_transform(
                    X_tr, y_train,
                    include_interactions=True,
                    include_polynomials=False,  # disabled: reduces overfit
                    include_binning=True,
                    select_features=True,
                )
                X_te = engineer.transform(X_te)  # only train-derived params applied
            
            prep = DataPreprocessor().fit(X_tr)
            X_tr_p, X_te_p = prep.transform(X_tr), prep.transform(X_te)
            
            for m_name in models_to_test:
                m_type = 'ensemble' if m_name in ['random_forest', 'xgboost', 'lightgbm'] else 'baseline'
                model = self.trainer.train_model(m_name, X_tr_p, y_train, m_type)
                
                y_pred = model.predict(X_te_p)
                p = self.evaluator.evaluate_model(model, X_te_p, y_test, m_name)
                f = self.fairness_eval.evaluate_fairness_fairlearn(y_test, y_pred, age_bins_test, m_name)
                
                # CI Bootstrapping (Limited to 50 for multi-dataset speed)
                ci = self.research_analyser.bootstrap_metrics(model, X_te_p, y_test, n_iterations=50)
                ci_str = f"[{ci.loc['roc_auc', 'ci_low']:.2f}, {ci.loc['roc_auc', 'ci_high']:.2f}]"
                
                perf_rows.append({
                    'Model': m_name, 'Feature Eng': fe_label, 
                    'Accuracy (%)': p['accuracy']*100, 'Precision': p['precision'],
                    'Recall': p['recall'], 'F1-Score': p['f1_score'], 
                    'ROC-AUC': p['roc_auc'], '95% CI': ci_str
                })
                
                fair_rows.append({
                    'Model': m_name, 'Feature Eng': fe_label,
                    'Demographic Parity Diff': f.get('demographic_parity_difference', 0),
                    'Equalized Odds Diff': f.get('equalized_odds_difference', 0),
                    'TPR Difference': f.get('tpr_difference', 0),
                    'FPR Difference': f.get('fpr_difference', 0)
                })
                
                # CV Stability (Fairness) - only for Yes FE as requested for Table 7
                if fe_enabled:
                    fs = self.research_analyser.bootstrap_fairness_metrics(model, X_te_p, y_test, age_bins_test, n_iterations=30)
                    cv_fair_rows.append({
                        'Model': f"{m_name.replace('_', ' ').title()} (Yes FE)",
                        'DP Diff (Mean ± SD)': f"{fs.loc['dp_diff', 'mean']:.3f} ± {fs.loc['dp_diff', 'std']:.3f}",
                        'EO Diff (Mean ± SD)': f"{fs.loc['eo_diff', 'mean']:.3f} ± {fs.loc['eo_diff', 'std']:.3f}"
                    })
                
                # CV
                cv = self.trainer.cross_validate_model(model, X_tr_p, y_train, cv=5)
                cv_rows.append({
                    'Model': m_name, 'Accuracy (Mean)': cv['mean_score'], 'FE': fe_label, 'scores': cv['scores']
                })
                
                all_models_trained[f"{m_name}_{fe_label}"] = model
                
                if fe_enabled and p['roc_auc'] > best_auc:
                    best_auc = p['roc_auc']
                    best_id = m_name
                    best_m = model
                    X_te_best_p = X_te_p
                    X_tr_best_p = X_tr_p
                
                if not fe_enabled:
                    X_te_base_p = X_te_p
                    X_tr_base_p = X_tr_p

        current_data['performance'] = pd.DataFrame(perf_rows)
        current_data['fairness'] = pd.DataFrame(fair_rows)
        current_data['cv_results'] = pd.DataFrame(cv_rows)
        current_data['cv_fairness'] = pd.DataFrame(cv_fair_rows)

        # 3. SUBGROUP
        print("-> Computing Subgroup Performance...")
        counts = age_bins_test.value_counts()
        print(f"   [Subgroup N-counts]:\n{counts}")
        for group, count in counts.items():
            if count < 20: 
                print(f"   ⚠️ WARNING: Group '{group}' has small sample size ({count}). Subgroup metrics may be unstable.")
                
        sg = self.clinical_eval.get_subgroup_metrics(y_test, best_m.predict(X_te_best_p), age_bins_test, best_id)
        current_data['subgroup'] = sg

        # 4. ABLATION
        current_data['ablation'] = self.run_ablation_study(dataset_name)

        # 5. MITIGATION (Enhanced with Reweighing + ThresholdOpt)
        print("-> Applying Multi-Step Bias Mitigation...")
        mit_rows = []
        for m_name in ['xgboost', 'random_forest']:
            key = f"{m_name}_Yes"
            if key in all_models_trained:
                m_obj = all_models_trained[key]
                
                # --- STRATEGY 1: ThresholdOptimizer (Equalized Odds) ---
                # BUG FIX: Equalized Odds usually better than DP for preventing correctness drop
                mit_model = self.mitigator.apply_mitigation('threshold_optimizer', m_obj, X_tr_best_p, y_train, age_bins_train, constraint='equalized_odds')
                y_pred_mit = mit_model.predict(X_te_best_p, sensitive_features=age_bins_test)
                
                f_after = self.fairness_eval.evaluate_fairness_fairlearn(y_test, y_pred_mit, age_bins_test, f"{m_name}_mit")
                f_before = current_data['fairness'][(current_data['fairness']['Model'] == m_name) & (current_data['fairness']['Feature Eng'] == 'Yes')]['Demographic Parity Diff'].values[0]
                f_after_val = f_after.get('demographic_parity_difference', 0)
                acc_before = current_data['performance'][(current_data['performance']['Model'] == m_name) & (current_data['performance']['Feature Eng'] == 'Yes')]['Accuracy (%)'].values[0] / 100
                acc_after = accuracy_score(y_test, y_pred_mit)
                
                mit_rows.append({
                    'Model': m_name, 'Strategy': 'Threshold Opt (EO)',
                    'Accuracy (Before)': acc_before, 'Accuracy (After)': acc_after,
                    'Δ Accuracy': acc_after - acc_before,
                    'DP Diff (Before)': f_before, 'DP Diff (After)': f_after_val,
                    'Δ Fairness': f_after_val - f_before
                })

                # --- STRATEGY 2: Reweighing (Preprocessing) ---
                try:
                    # BUG FIX: AIF360 Reweighing requires a BINARY/CATEGORICAL sensitive
                    # attribute, not the raw continuous Age column. Using the binned
                    # sensitive column (age_bins_train) converted to a numeric indicator.
                    # Old code passed sensitive_col='Age' which is continuous (21,45,62,...)
                    # causing AIF360 to treat every age as a distinct protected class.
                    rew_col = '_rew_sensitive'
                    age_bins_train_arr = self.get_sensitive_bins(X_train, dataset_name)
                    X_rew_input = X_tr_best_p.copy()
                    X_rew_input[rew_col] = (age_bins_train_arr.values == age_bins_train_arr.cat.categories[-1]).astype(int)

                    privileged   = [{rew_col: 1}]  # oldest bin = privileged
                    unprivileged = [{rew_col: 0}]  # younger bins = unprivileged

                    X_re, y_re, weights = self.mitigator.reweighing(
                        X_rew_input, y_train, rew_col, privileged, unprivileged
                    )
                    X_re = X_re.drop(columns=[rew_col], errors='ignore')

                    # Retrain with weights
                    m_rew = self.trainer.train_model(m_name, X_re, y_re, 'ensemble')
                    m_rew.fit(X_re, y_re, sample_weight=weights)
                    
                    y_pred_rew = m_rew.predict(X_te_best_p)
                    f_rew = self.fairness_eval.evaluate_fairness_fairlearn(y_test, y_pred_rew, age_bins_test, f"{m_name}_rew")
                    f_rew_val = f_rew.get('demographic_parity_difference', 0)
                    acc_rew = accuracy_score(y_test, y_pred_rew)
                    
                    mit_rows.append({
                        'Model': m_name, 'Strategy': 'AIF360 Reweighing',
                        'Accuracy (Before)': acc_before, 'Accuracy (After)': acc_rew,
                        'Δ Accuracy': acc_rew - acc_before,
                        'DP Diff (Before)': f_before, 'DP Diff (After)': f_rew_val,
                        'Δ Fairness': f_rew_val - f_before
                    })
                except Exception as e:
                    print(f"   ⚠️ Reweighing failed for {m_name}: {e}")
        
        current_data['mitigation'] = pd.DataFrame(mit_rows)

        # 5b. COMPLEXITY
        current_data['complexity'] = self.run_complexity_fairness_analysis(X_tr_best_p, X_te_best_p, y_train, y_test, age_bins_test)

        # 6. SIGNIFICANCE
        print("-> Running Significance Testing (Fairness Impact)...")
        sig_rows = []
        # Target specific models mentioned in Issue 3 requirements
        target_models = ['xgboost', 'decision_tree', 'naive_bayes']
        
        for m_name in target_models:
            base_key, fe_key = f"{m_name}_No", f"{m_name}_Yes"
            if base_key in all_models_trained and fe_key in all_models_trained:
                # Generate local distributions of fairness metrics for comparison
                # Use X_te_base_p for the baseline model (No FE)
                base_dist = self.research_analyser.bootstrap_fairness_metrics(
                    all_models_trained[base_key], X_te_base_p, y_test, age_bins_test, 
                    n_iterations=50, return_samples=True
                )['dp_diff'].values
                
                # Use X_te_best_p for the engineered model (Yes FE)
                fe_dist = self.research_analyser.bootstrap_fairness_metrics(
                    all_models_trained[fe_key], X_te_best_p, y_test, age_bins_test, 
                    n_iterations=50, return_samples=True
                )['dp_diff'].values
                
                # Test for significance in bias change (lower is better)
                sig = self.research_analyser.statistical_significance_test(base_dist, fe_dist, higher_is_better=False)
                
                sig_rows.append({
                    'Comparison': f"{m_name}: No FE vs Yes FE", 
                    'Metric': 'DP Diff',
                    'p_value': sig['p_value'], 
                    'Cohen\'s d': sig['cohens_d'], 
                    'Significant?': "✅ YES" if sig['significant'] else "❌ NO",
                    'Interpretation': sig['interpretation']
                })
        current_data['significance'] = pd.DataFrame(sig_rows)

        # Visualizations
        vis_dir = self.output_root / dataset_name / "graphs"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 7. EXPLAINABILITY (SHAP)
        print("-> Running SHAP Explainability...")
        try:
            self.shap_analyzer.create_explainer(best_m, X_tr_best_p, model_name=best_id)
            self.shap_analyzer.plot_summary(best_id, X_te_best_p, save_path=vis_dir / "G6_SHAP_Summary.png")
        except Exception as e:
            print(f"   ⚠️ SHAP Analysis failed: {e}")

        self.paper_vis.plot_roc_curves({k:v for k,v in all_models_trained.items() if 'Yes' in k}, X_te_best_p, y_test, save_path=vis_dir / "G1_ROC.png")
        
        # G2: Pareto Frontier
        pareto_ready_df = pd.merge(current_data['performance'], current_data['fairness'], on=['Model', 'Feature Eng'])
        pareto_opt = self.research_analyser.calculate_pareto_frontier(pareto_ready_df, 'ROC-AUC', 'Demographic Parity Diff')
        self.paper_vis.plot_pareto_frontier(pareto_ready_df, pareto_opt, 'ROC-AUC', 'Demographic Parity Diff', save_path=vis_dir / "G2_Pareto.png")
        
        self.paper_vis.plot_fairness_bar_chart(current_data['fairness'][current_data['fairness']['Feature Eng'] == 'Yes'], save_path=vis_dir / "G3_Fairness_Bar.png")
        self.paper_vis.plot_subgroup_heatmap(sg, save_path=vis_dir / "G4_Subgroup_Heatmap.png")
        self.paper_vis.plot_mitigation_impact(current_data['mitigation'], save_path=vis_dir / "G9_Mitigation.png")
        self.paper_vis.plot_significance_heatmap(current_data['significance'], save_path=vis_dir / "G10_Significance.png")
        
        # G11: CV Boxplots
        cv_scores = [row['scores'] for row in cv_rows if row['FE'] == 'Yes']
        cv_labels = [row['Model'] for row in cv_rows if row['FE'] == 'Yes']
        self.paper_vis.plot_cv_boxplots(cv_scores, cv_labels, save_path=vis_dir / "G11_CV_Boxplots.png")
        
        self.paper_vis.plot_calibration_curve_subgroups(best_m, X_te_best_p, y_test, age_bins_test, save_path=vis_dir / "G12_Calibration.png")
        self.paper_vis.plot_fairness_confusion_matrices(y_test, best_m.predict(X_te_best_p), age_bins_test, save_path=vis_dir / "G13_Confusion_Matrix.png")
        self.paper_vis.plot_complexity_vs_fairness(current_data['complexity'], save_path=vis_dir / "G14_Complexity.png")

        return current_data

    def run_all(self):
        """Unified runner for publication assets."""
        self.run_publication_pipeline()
