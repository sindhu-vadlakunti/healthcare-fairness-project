"""Bias mitigation utilities using Fairlearn and AIF360."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.base import BaseEstimator

# Fairlearn imports
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# AIF360 imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    import tensorflow as tf
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("Warning: AIF360 or TensorFlow not available. Some mitigation techniques will be unavailable.")


class BiasMitigator:
    """Mitigate bias in machine learning models."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize bias mitigator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.mitigated_models = {}
        self.reweighing_transformer = None
    
    def threshold_optimization(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        constraint: str = 'demographic_parity'
    ) -> ThresholdOptimizer:
        """
        Apply threshold optimization for fairness.
        
        Args:
            estimator: Base estimator
            X_train: Training features
            y_train: Training labels
            sensitive_features: Sensitive attributes
            constraint: Fairness constraint ('demographic_parity' or 'equalized_odds')
            
        Returns:
            Threshold-optimized model
        """
        print(f"Applying threshold optimization with {constraint} constraint...")
        
        # Create threshold optimizer
        threshold_optimizer = ThresholdOptimizer(
            estimator=estimator,
            constraints=constraint,
            predict_method='predict_proba'
        )
        
        # Fit the optimizer
        threshold_optimizer.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        return threshold_optimizer
    
    def exponentiated_gradient(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        constraint: str = 'demographic_parity'
    ) -> ExponentiatedGradient:
        """
        Apply Exponentiated Gradient reduction for fairness.
        
        Args:
            estimator: Base estimator
            X_train: Training features
            y_train: Training labels
            sensitive_features: Sensitive attributes
            constraint: Fairness constraint
            
        Returns:
            Fair model using Exponentiated Gradient
        """
        print(f"Applying Exponentiated Gradient with {constraint} constraint...")
        
        # Select constraint
        if constraint == 'demographic_parity':
            fairness_constraint = DemographicParity()
        elif constraint == 'equalized_odds':
            fairness_constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unknown constraint: {constraint}")
        
        # Create Exponentiated Gradient reducer
        mitigator = ExponentiatedGradient(
            estimator=estimator,
            constraints=fairness_constraint
        )
        
        # Fit the mitigator
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        return mitigator
    
    def reweighing(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_attribute: str,
        privileged_groups: List[Dict],
        unprivileged_groups: List[Dict]
    ) -> tuple:
        """
        Apply reweighing preprocessing technique.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_attribute: Name of sensitive attribute
            privileged_groups: List of privileged group definitions
            unprivileged_groups: List of unprivileged group definitions
            
        Returns:
            Tuple of (transformed_X, transformed_y, sample_weights)
        """
        if not AIF360_AVAILABLE:
            print("AIF360 not available. Cannot apply reweighing.")
            return X_train, y_train, None
        
        print("Applying reweighing technique...")
        
        try:
            # Prepare data for AIF360
            df = X_train.copy()
            df['label'] = y_train.values
            
            # Create BinaryLabelDataset
            dataset = BinaryLabelDataset(
                df=df,
                label_names=['label'],
                protected_attribute_names=[sensitive_attribute]
            )
            
            # Apply reweighing
            reweigher = Reweighing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )
            
            dataset_transformed = reweigher.fit_transform(dataset)
            
            # Extract weights
            sample_weights = dataset_transformed.instance_weights
            
            self.reweighing_transformer = reweigher
            
            return X_train, y_train, sample_weights
            
        except Exception as e:
            print(f"Error applying reweighing: {e}")
            return X_train, y_train, None
    
    def adversarial_debiasing(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_attribute: str,
        privileged_groups: List[Dict],
        unprivileged_groups: List[Dict],
        num_epochs: int = 50
    ) -> Any:
        """
        Apply adversarial debiasing in-processing technique.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_attribute: Name of sensitive attribute
            privileged_groups: List of privileged group definitions
            unprivileged_groups: List of unprivileged group definitions
            num_epochs: Number of training epochs
            
        Returns:
            Trained adversarial debiasing model
        """
        if not AIF360_AVAILABLE:
            print("AIF360 or TensorFlow not available. Cannot apply adversarial debiasing.")
            return None
        
        print("Applying adversarial debiasing...")
        
        try:
            # Prepare data for AIF360
            df = X_train.copy()
            df['label'] = y_train.values
            
            # Create BinaryLabelDataset
            dataset = BinaryLabelDataset(
                df=df,
                label_names=['label'],
                protected_attribute_names=[sensitive_attribute]
            )
            
            # Reset TensorFlow session
            tf.compat.v1.reset_default_graph()
            sess = tf.compat.v1.Session()
            
            # Apply adversarial debiasing
            debiaser = AdversarialDebiasing(
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups,
                scope_name='debiaser',
                debias=True,
                sess=sess,
                num_epochs=num_epochs,
                batch_size=128
            )
            
            debiaser.fit(dataset)
            
            return debiaser
            
        except Exception as e:
            print(f"Error applying adversarial debiasing: {e}")
            return None
    
    def apply_mitigation(
        self,
        technique: str,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        **kwargs
    ) -> Any:
        """
        Apply specified bias mitigation technique.
        
        Args:
            technique: Mitigation technique name
            estimator: Base estimator
            X_train: Training features
            y_train: Training labels
            sensitive_features: Sensitive attributes
            **kwargs: Additional arguments for specific techniques
            
        Returns:
            Mitigated model or transformed data
        """
        if technique == 'threshold_optimizer':
            return self.threshold_optimization(
                estimator, X_train, y_train, sensitive_features,
                constraint=kwargs.get('constraint', 'demographic_parity')
            )
        
        elif technique == 'exponentiated_gradient':
            return self.exponentiated_gradient(
                estimator, X_train, y_train, sensitive_features,
                constraint=kwargs.get('constraint', 'demographic_parity')
            )
        
        elif technique == 'reweighing':
            return self.reweighing(
                X_train, y_train,
                sensitive_attribute=kwargs.get('sensitive_attribute'),
                privileged_groups=kwargs.get('privileged_groups'),
                unprivileged_groups=kwargs.get('unprivileged_groups')
            )
        
        elif technique == 'adversarial_debiasing':
            return self.adversarial_debiasing(
                X_train, y_train,
                sensitive_attribute=kwargs.get('sensitive_attribute'),
                privileged_groups=kwargs.get('privileged_groups'),
                unprivileged_groups=kwargs.get('unprivileged_groups'),
                num_epochs=kwargs.get('num_epochs', 50)
            )
        
        else:
            raise ValueError(f"Unknown mitigation technique: {technique}")
    
    def compare_before_after_mitigation(
        self,
        fairness_metrics_before: Dict[str, float],
        fairness_metrics_after: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compare fairness metrics before and after mitigation.
        
        Args:
            fairness_metrics_before: Fairness metrics before mitigation
            fairness_metrics_after: Fairness metrics after mitigation
            
        Returns:
            DataFrame comparing metrics
        """
        comparison = pd.DataFrame({
            'Before Mitigation': fairness_metrics_before,
            'After Mitigation': fairness_metrics_after
        })
        
        comparison['Improvement'] = (
            comparison['Before Mitigation'].abs() - 
            comparison['After Mitigation'].abs()
        )
        
        return comparison
