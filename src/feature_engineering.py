"""Feature engineering utilities - stateful fit/transform to prevent data leakage."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Optional
from itertools import combinations


class FeatureEngineer:
    """
    Create and engineer features for machine learning.

    CRITICAL DESIGN: This class follows a strict fit/transform pattern.
    - fit_transform(X_train, y_train) : learns ALL bin edges, poly shapes,
      selected feature names FROM TRAINING DATA ONLY.
    - transform(X_test)              : applies the exact same learned parameters
      to the test data. No information from X_test is ever used.
    This design completely eliminates data leakage.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        # --- State learned during fit ---
        self._fitted = False
        self._interaction_pairs: List[Tuple[str, str]] = []
        self._binners: dict = {}          # {feature_name: KBinsDiscretizer}
        self._bin_feature_names: List[str] = []
        self._poly_transformer: Optional[PolynomialFeatures] = None
        self._poly_numeric_cols: List[str] = []
        self._selected_features: Optional[List[str]] = None
        # For ablation compatibility (legacy single-call methods)
        self.interaction_features = []
        self.polynomial_transformer = None
        self.feature_selector = None
        self.binning_info = {}

    # ------------------------------------------------------------------
    # PRIMARY API: stateful fit_transform / transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        include_interactions: bool = True,
        include_polynomials: bool = False,
        include_binning: bool = True,
        polynomial_degree: int = 2,
        n_bins: int = 5,
        select_features: bool = True,
        n_features_to_select: int = 20,
    ) -> pd.DataFrame:
        """
        Fit ALL transformers on X (training data only) and return transformed X.
        Call transform() on the test set afterwards.
        """
        X_out = X.copy()

        # --- Interactions ---
        if include_interactions:
            print("  -> [FE] Fitting interaction terms on TRAIN only...")
            numeric_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 20:
                self._interaction_pairs = list(combinations(numeric_cols[:10], 2))
            else:
                self._interaction_pairs = list(combinations(numeric_cols, 2))
            for f1, f2 in self._interaction_pairs:
                X_out[f"{f1}_x_{f2}"] = X_out[f1] * X_out[f2]

        # --- Binning (fit on train, no leakage) ---
        if include_binning:
            print("  -> [FE] Fitting binners on TRAIN only...")
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self._bin_feature_names = []
            for feat in numeric_cols:
                binner = KBinsDiscretizer(
                    n_bins=n_bins, encode='ordinal', strategy='quantile',
                    subsample=None
                )
                binner.fit(X[[feat]])
                self._binners[feat] = binner
                bin_name = f"{feat}_binned"
                X_out[bin_name] = binner.transform(X[[feat]]).flatten().astype(int)
                self._bin_feature_names.append(bin_name)

        # --- Polynomial (fit on train, no leakage) ---
        if include_polynomials:
            print("  -> [FE] Fitting polynomial transformer on TRAIN only...")
            self._poly_numeric_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
            self._poly_transformer = PolynomialFeatures(
                degree=polynomial_degree, include_bias=False, interaction_only=False
            )
            poly_arr = self._poly_transformer.fit_transform(X_out[self._poly_numeric_cols])
            poly_names = self._poly_transformer.get_feature_names_out(self._poly_numeric_cols)
            X_out = pd.DataFrame(poly_arr, columns=poly_names, index=X_out.index)

        # --- Feature selection (correlation with TARGET, train only) ---
        if select_features:
            X_numeric = X_out.select_dtypes(include=[np.number])
            if X_numeric.shape[1] > n_features_to_select:
                print(f"  -> [FE] Selecting top {n_features_to_select} features "
                      f"from {X_numeric.shape[1]} columns (train correlations)...")
                # Ensure index alignment
                correlations = X_numeric.corrwith(pd.Series(y.values, index=X_numeric.index)).abs()
                self._selected_features = correlations.nlargest(n_features_to_select).index.tolist()
            else:
                self._selected_features = X_numeric.columns.tolist()
            X_out = X_out[self._selected_features]

        self._fitted = True
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the SAME transformations learned from training data to any new set.
        This is the only correct way to transform test data - zero leakage.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureEngineer.transform() called before fit_transform(). "
                "Always call fit_transform(X_train, y_train) first."
            )
        X_out = X.copy()

        # --- Apply learned interactions ---
        for f1, f2 in self._interaction_pairs:
            if f1 in X_out.columns and f2 in X_out.columns:
                X_out[f"{f1}_x_{f2}"] = X_out[f1] * X_out[f2]

        # --- Apply learned binners (uses TRAIN edges, not test distribution) ---
        for feat, binner in self._binners.items():
            if feat in X_out.columns:
                bin_name = f"{feat}_binned"
                X_out[bin_name] = binner.transform(X_out[[feat]]).flatten().astype(int)

        # --- Apply learned polynomial transformer ---
        if self._poly_transformer is not None:
            available = [c for c in self._poly_numeric_cols if c in X_out.columns]
            missing = [c for c in self._poly_numeric_cols if c not in X_out.columns]
            X_poly_input = X_out[available].copy()
            for c in missing:
                X_poly_input[c] = 0.0
            X_poly_input = X_poly_input[self._poly_numeric_cols]
            poly_arr = self._poly_transformer.transform(X_poly_input)
            poly_names = self._poly_transformer.get_feature_names_out(self._poly_numeric_cols)
            X_out = pd.DataFrame(poly_arr, columns=poly_names, index=X_out.index)

        # --- Apply learned feature selection ---
        if self._selected_features is not None:
            available = [c for c in self._selected_features if c in X_out.columns]
            for c in self._selected_features:
                if c not in X_out.columns:
                    X_out[c] = 0.0              # fill only truly absent columns
            X_out = X_out[self._selected_features]

        return X_out

    # ------------------------------------------------------------------
    # LEGACY / ABLATION METHODS (kept for compatibility with ablation loop)
    # These are fine for ablation where train==test column structure.
    # ------------------------------------------------------------------

    def create_interaction_terms(
        self,
        X: pd.DataFrame,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        X_new = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if feature_pairs is None:
            if len(numeric_cols) > 20:
                feature_pairs = list(combinations(numeric_cols[:10], 2))
            else:
                feature_pairs = list(combinations(numeric_cols, 2))
        for feat1, feat2 in feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                name = f"{feat1}_x_{feat2}"
                X_new[name] = X[feat1] * X[feat2]
                self.interaction_features.append(name)
        return X_new

    def create_polynomial_features(
        self, X: pd.DataFrame, degree: int = 2, include_bias: bool = False
    ) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return X
        self.polynomial_transformer = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=False
        )
        X_poly = self.polynomial_transformer.fit_transform(X[numeric_cols])
        feature_names = self.polynomial_transformer.get_feature_names_out(numeric_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        for col in [c for c in X.columns if c not in numeric_cols]:
            X_poly_df[col] = X[col].values
        return X_poly_df

    def create_binned_features(
        self,
        X: pd.DataFrame,
        features_to_bin: Optional[List[str]] = None,
        strategy: str = 'quantile',
        n_bins: int = 5
    ) -> pd.DataFrame:
        X_new = X.copy()
        if features_to_bin is None:
            features_to_bin = X.select_dtypes(include=[np.number]).columns.tolist()
        for feature in features_to_bin:
            if feature not in X.columns:
                continue
            bin_name = f"{feature}_binned"
            binner = KBinsDiscretizer(
                n_bins=n_bins, encode='ordinal', strategy=strategy,
                subsample=None
            )
            X_new[bin_name] = binner.fit_transform(X[[feature]]).flatten().astype(int)
            self.binning_info[feature] = {'strategy': strategy, 'n_bins': n_bins, 'bin_name': bin_name}
        return X_new

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        method: str = 'correlation'
    ) -> pd.DataFrame:
        X_numeric = X.select_dtypes(include=[np.number])
        if method == 'rfe':
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            self.feature_selector = RFE(
                estimator=estimator,
                n_features_to_select=min(n_features, X_numeric.shape[1])
            )
            self.feature_selector.fit(X_numeric, y)
            selected = X_numeric.columns[self.feature_selector.support_].tolist()
        elif method == 'variance':
            variances = X_numeric.var()
            selected = variances.nlargest(min(n_features, len(variances))).index.tolist()
        elif method == 'correlation':
            correlations = X_numeric.corrwith(pd.Series(y.values, index=X_numeric.index)).abs()
            selected = correlations.nlargest(min(n_features, len(correlations))).index.tolist()
        else:
            raise ValueError(f"Unknown selection method: {method}")
        return X[selected]

    def engineer_all_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        include_interactions: bool = True,
        include_polynomials: bool = False,
        include_binning: bool = True,
        polynomial_degree: int = 2,
        n_bins: int = 5,
        select_features: bool = False,
        n_features_to_select: int = 20
    ) -> pd.DataFrame:
        """Legacy method - used ONLY inside the ablation loop (train-only context)."""
        X_engineered = X.copy()
        if include_interactions:
            X_engineered = self.create_interaction_terms(X_engineered)
        if include_binning:
            X_engineered = self.create_binned_features(X_engineered, n_bins=n_bins)
        if include_polynomials:
            X_engineered = self.create_polynomial_features(X_engineered, degree=polynomial_degree)
        if select_features and y is not None:
            if X_engineered.shape[1] > n_features_to_select:
                X_engineered = self.select_features(X_engineered, y, n_features=n_features_to_select)
        return X_engineered
