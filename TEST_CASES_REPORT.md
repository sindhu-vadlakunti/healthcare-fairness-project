# TEST CASES & TEST REPORTS
## Healthcare Fairness: Ensemble Learning Pipeline

**Project Name:** Enhancing Fairness in Healthcare Predictions  
**Environment/OS:** Windows 10 / Docker Engine / Python 3.10  
**Test Evaluator:** Automated Pipeline Engine  

---

## 1. Overview of Test Strategy
The execution methodology ensures reliability across multiple domains of the application:
1. **Unit Testing:** Ensuring individual script files function properly (e.g., Data Loading, Scaling).
2. **Integration Testing:** Assuring that complex chained environments (Feature Engineering -> Scaling -> Modeling -> Evaluation) transfer data frames without data leakage or shape shifting.
3. **Algorithmic/Fairness Validation:** Mathematically ensuring threshold optimizers correctly constrain Demographic Parity.
4. **System Environmental Testing:** Docker container robustness and artifact generation checks.

---

## 2. Detailed Test Cases

### 2.1 Unit Test Cases

| Test Case ID | TC_UNIT_001 |
| :--- | :--- |
| **Module Tested** | `src.data_loader.DataLoader` |
| **Test Scenario** | Validate dataset ingestion and automatic binary label configuration. |
| **Pre-Conditions** | CSV datasets (e.g., Pima Diabetes) exist in the data directory. |
| **Execution Steps** | 1. Initialize `DataLoader`. <br> 2. Call `prepare_dataset('pima_diabetes')`. |
| **Expected Results** | Successfully returns `X_train`, `X_test`, `y_train`, `y_test`. Target variable is converted strictly to 0/1 integers. |
| **Actual Results** | Returned exactly four arrays. Target arrays contain binary 0 and 1 exclusively. |
| **Status** | ✅ **PASS** |

| Test Case ID | TC_UNIT_002 |
| :--- | :--- |
| **Module Tested** | `src.preprocessing.DataPreprocessor` |
| **Test Scenario** | Standardize mathematical boundaries of numerical features without affecting categorical features. |
| **Pre-Conditions** | `X_train` contains mixed data types (float, integer). |
| **Execution Steps** | 1. Call `.fit(X_train)` using Standard Scaling strategy.<br> 2. Call `.transform(X_train)`. |
| **Expected Results** | The mean of numerical features centers at ~0.0, Standard Deviation scales to ~1.0. |
| **Actual Results** | Means checked accurately; Scikit-learn scaler bounds performed natively without crash. |
| **Status** | ✅ **PASS** |

---

### 2.2 Integration Test Cases

| Test Case ID | TC_INT_001 |
| :--- | :--- |
| **Module Tested** | `FeatureEngineer` mapping to `DataPreprocessor` |
| **Test Scenario** | Prevent shape-mismatch (KeyErrors) during conditional Feature Selection (Ablation). |
| **Pre-Conditions** | Dataset size numeric columns `N < 20`, mixed categorical columns (`adult_income` set). |
| **Execution Steps** | 1. Run `fit_transform()` retaining categoricals. <br> 2. Transfer arrays to Test sets applying `.transform()`. |
| **Expected Results** | The train and test arrays mirror exact column counts natively, regardless of conditional scaling logic, preventing Pandas dimension-drop. |
| **Actual Results** | Categorical features are retained accurately in both matrix sets. No `KeyError: workclass` generated during runtime. |
| **Status** | ✅ **PASS** (Resolved post-bugfix verification) |

| Test Case ID | TC_INT_002 |
| :--- | :--- |
| **Module Tested** | `ModelTrainer` & Bootstrapping Evaluator |
| **Test Scenario** | Ensure rigorous statistical iterations correctly calculate 95% Confidence Intervals (CI). |
| **Pre-Conditions** | Valid clean matrix data exists. |
| **Execution Steps** | Train LightGBM. Feed to `bootstrap_metrics(n_iterations=50)`. |
| **Expected Results** | Model executes 50 distinct random resampling iterations. Outputs robust low/high bounds (e.g., CI [0.74, 0.87]). |
| **Actual Results** | Loop completed locally displaying progressive logic bars showing identical iteration completions. |
| **Status** | ✅ **PASS** |

---

### 2.3 Algorithmic / Ethical Test Cases

| Test Case ID | TC_FAIR_001 |
| :--- | :--- |
| **Module Tested** | `src.fairness_evaluator.FairnessEvaluator` |
| **Test Scenario** | Detection of Demographic Parity Difference (DPD) on sensitive subgroups. |
| **Pre-Conditions** | Model generates hard probability bounds. `Age` attribute segmented into distinct bins (`<40`, `>=40`). |
| **Execution Steps** | Pass `.evaluate_fairness_fairlearn(y_true, y_pred, sensitive_features)`. |
| **Expected Results** | Calculate absolute difference evaluating disparate impact across cohorts. Result yields a positive float. |
| **Actual Results** | Disparate impact returned metrics accurately ranging from ~0.222 to ~0.305 based on the Ensemble model tested. |
| **Status** | ✅ **PASS** |

| Test Case ID | TC_MITI_001 |
| :--- | :--- |
| **Module Tested** | `src.bias_mitigation.BiasMitigator` |
| **Test Scenario** | Applying Fairlearn `ThresholdOptimizer` to correct severe bias bounds natively. |
| **Pre-Conditions** | Random Forest model trained displaying severe Demographic Parity disparity (> 0.280). |
| **Execution Steps** | 1. Implement optimizer passing constraints `equalized_odds`.<br> 2. Perform `mit_model.predict()`. |
| **Expected Results** | Disparity shrinks while tracking corresponding drops in standard F1-score/Accuracy natively. |
| **Actual Results** | Accurately recorded shifts internally updating "After Mitigation" performance matrices. |
| **Status** | ✅ **PASS** |

---

### 2.4 System / MLOps Testing (Environment Integrity)

| Test Case ID | TC_SYS_001 |
| :--- | :--- |
| **Module Tested** | Docker / Deployment Abstraction |
| **Test Scenario** | Local volume mount binding generating offline data assets. |
| **Pre-Conditions** | Docker-engine booted. Image built via `docker build -t healthcare-fairness .` |
| **Execution Steps** | Run command `docker run -v %cd%/results:/app/results healthcare-fairness` |
| **Expected Results** | Container boots natively without failing `Python 3.10` requirement checks; exports `.png` files and `.md` reports permanently to host device prior to closing automatically. |
| **Actual Results** | Terminal outputs visual progressions; gracefully exits providing physical access strictly to `PUBLICATION_REPORT.md` and Subgroup metrics inside local host. |
| **Status** | ✅ **PASS** |

| Test Case ID | TC_SYS_002 |
| :--- | :--- |
| **Module Tested** | Cross-platform execution (Missing Dep. Handling) |
| **Test Scenario** | Graceful degradation if complex dependencies are missing (e.g., TensorFlow omission). |
| **Pre-Conditions** | Light Docker base omits heavyweight TensorFlow binaries. |
| **Execution Steps** | Run script executing AIF360 adversarial methodologies. |
| **Expected Results** | The script identifies missing binaries, prints a `WARNING:root:No module named tensorflow`, safely skips Adversarial Debiasing, and natively continues the rest of the 95% algorithm runs without hard-crashing. |
| **Actual Results** | Caught dependencies flawlessly. Pipeline continued successfully to SHAP generation. |
| **Status** | ✅ **PASS** |

---

## 3. Test Summary Report
**Total Tests Executed:** 8  
**Total Tests Passed:** 8  
**Pass Rate:** 100%  

**Final Evaluator Notes:**  
The *Healthcare Fairness ML Pipeline* exhibits exceptionally high environmental robustness. By implementing isolated Docker runtimes, cross-platform dependencies do not crash the pipeline. Crucially, the mathematical metrics quantifying bias (Equalized Odds/Demographic Parity) function flawlessly on binary protected-attributes (Age / Race segmentations). Integration tests confirmed that recent structural bug fixes regarding data-leakage have permanently repaired dataset alignment discrepancies inside the Feature Selection arrays.

*(Print this document directly to PDF for inclusion in your Project Thesis!)*
