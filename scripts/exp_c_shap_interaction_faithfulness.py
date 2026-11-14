"""
Experiment C: 2nd-Order SHAP Interaction Synergy & Explanation Faithfulness
Evaluates:
- 2nd-Order Pairwise TreeSHAP Interaction Matrix Phi_{i,j}
- Empirical proof of Non-linear Synergy: (Projects x Communication) > 0 and (Projects x CGPA) Compensatory Offset
- Attribution Faithfulness Audit: Deletion & Insertion Curves vs. Random Baselines
"""

import os
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
DATA_PATH = os.path.join(RAW_DIR, "kaggle", "college-student-placement", "college_student_placement_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "derived_matrices")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Exp C] Loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df['Placement_Binary'] = (df['Placement'] == 'Yes').astype(int)
df['Internship_Binary'] = (df['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df[feature_cols].values
y = df['Placement_Binary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"[Exp C] Training LightGBM on {len(X_train)} samples...")
model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=SEED, verbose=-1)
model.fit(X_train, y_train)

test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"[Exp C] Model Test ROC-AUC: {test_auc:.4f}")

# 1. Compute TreeSHAP Interaction Values
print("[Exp C] Computing TreeSHAP 2nd-order interaction values on sample N=1000...")
explainer = shap.TreeExplainer(model)
sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
X_sample = X_test[sample_idx]

# shap_interaction_values has shape (N, D, D) or list for binary classification
shap_interaction = explainer.shap_interaction_values(X_sample)
if isinstance(shap_interaction, list):
    # For binary classification, take positive class
    shap_interaction = shap_interaction[1]
elif shap_interaction.ndim == 4:
    shap_interaction = shap_interaction[:, :, :, 1]

# Compute Mean Absolute Interaction Matrix (8x8)
mean_abs_interaction = np.mean(np.abs(shap_interaction), axis=0) # (8, 8)
# Compute Signed Mean Interaction Matrix (8x8)
mean_signed_interaction = np.mean(shap_interaction, axis=0)

df_inter_abs = pd.DataFrame(mean_abs_interaction, index=feature_cols, columns=feature_cols)
df_inter_abs.to_csv(os.path.join(OUT_DIR, "table_exp_c_shap_interaction_matrix_abs.csv"))

df_inter_signed = pd.DataFrame(mean_signed_interaction, index=feature_cols, columns=feature_cols)
df_inter_signed.to_csv(os.path.join(OUT_DIR, "table_exp_c_shap_interaction_matrix_signed.csv"))

print("\n[Exp C] Top Pairwise SHAP Synergies (Mean Abs Interaction):")
pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        pairs.append({
            'Feature_1': feature_cols[i],
            'Feature_2': feature_cols[j],
            'Mean_Abs_Interaction': round(float(mean_abs_interaction[i, j]), 5),
            'Mean_Signed_Interaction': round(float(mean_signed_interaction[i, j]), 5)
        })
df_pairs = pd.DataFrame(pairs).sort_values('Mean_Abs_Interaction', ascending=False)
df_pairs.to_csv(os.path.join(OUT_DIR, "table_exp_c_shap_pairwise_ranking.csv"), index=False)
print(df_pairs.head(8))

# 2. Attribution Faithfulness Audit (Deletion & Insertion Curves)
print("\n[Exp C] Benchmarking Attribution Faithfulness (Deletion Curve vs. Random)...")
# Global feature importance from SHAP
mean_shap_importance = np.mean(np.abs(explainer.shap_values(X_sample)), axis=0)
if isinstance(mean_shap_importance, list):
    mean_shap_importance = mean_shap_importance[1]
elif mean_shap_importance.ndim == 2:
    mean_shap_importance = np.mean(mean_shap_importance, axis=0)

ranked_features_shap = np.argsort(-mean_shap_importance)
baseline_medians = np.median(X_train, axis=0)

# Deletion Evaluation
orig_probs = model.predict_proba(X_sample)[:, 1]

steps = list(range(len(feature_cols) + 1))
shap_deletion_errors = [0.0]
random_deletion_errors = [0.0]

for k in range(1, len(feature_cols) + 1):
    # SHAP Most-to-Least Deletion
    X_del_shap = X_sample.copy()
    for f in ranked_features_shap[:k]:
        X_del_shap[:, f] = baseline_medians[f]
    probs_del_shap = model.predict_proba(X_del_shap)[:, 1]
    shap_deletion_errors.append(np.mean(np.abs(orig_probs - probs_del_shap)))
    
    # Random Deletion (Average over 20 random permutations)
    rand_errs = []
    for _ in range(20):
        rand_f_idx = np.random.choice(len(feature_cols), size=k, replace=False)
        X_del_rand = X_sample.copy()
        for f in rand_f_idx:
            X_del_rand[:, f] = baseline_medians[f]
        probs_del_rand = model.predict_proba(X_del_rand)[:, 1]
        rand_errs.append(np.mean(np.abs(orig_probs - probs_del_rand)))
    random_deletion_errors.append(np.mean(rand_errs))

# Insertion Evaluation
shap_insertion_errors = [np.mean(np.abs(orig_probs - model.predict_proba(np.tile(baseline_medians, (len(X_sample), 1)))[:, 1]))]
random_insertion_errors = [shap_insertion_errors[0]]

for k in range(1, len(feature_cols) + 1):
    # SHAP Insertion (Starting from baseline, insert top k features)
    X_ins_shap = np.tile(baseline_medians, (len(X_sample), 1))
    for f in ranked_features_shap[:k]:
        X_ins_shap[:, f] = X_sample[:, f]
    probs_ins_shap = model.predict_proba(X_ins_shap)[:, 1]
    shap_insertion_errors.append(np.mean(np.abs(orig_probs - probs_ins_shap)))
    
    # Random Insertion
    rand_ins_errs = []
    for _ in range(20):
        rand_f_idx = np.random.choice(len(feature_cols), size=k, replace=False)
        X_ins_rand = np.tile(baseline_medians, (len(X_sample), 1))
        for f in rand_f_idx:
            X_ins_rand[:, f] = X_sample[:, f]
        probs_ins_rand = model.predict_proba(X_ins_rand)[:, 1]
        rand_ins_errs.append(np.mean(np.abs(orig_probs - probs_ins_rand)))
    random_insertion_errors.append(np.mean(rand_ins_errs))

# Compute Faithfulness AUC (trapezoidal rule)
deletion_auc_shap = np.trapezoid(shap_deletion_errors, dx=1.0) / len(feature_cols)
deletion_auc_rand = np.trapezoid(random_deletion_errors, dx=1.0) / len(feature_cols)

# For insertion, error should drop quickly, so lower insertion error AUC = more faithful
insertion_auc_shap = np.trapezoid(shap_insertion_errors, dx=1.0) / len(feature_cols)
insertion_auc_rand = np.trapezoid(random_insertion_errors, dx=1.0) / len(feature_cols)

faithfulness_gap = deletion_auc_shap - deletion_auc_rand

df_curves = pd.DataFrame({
    'Features_Modified': steps,
    'SHAP_Deletion_Prediction_Drop': [round(float(x), 4) for x in shap_deletion_errors],
    'Random_Deletion_Prediction_Drop': [round(float(x), 4) for x in random_deletion_errors],
    'SHAP_Insertion_Residual_Error': [round(float(x), 4) for x in shap_insertion_errors],
    'Random_Insertion_Residual_Error': [round(float(x), 4) for x in random_insertion_errors]
})
df_curves.to_csv(os.path.join(OUT_DIR, "table_exp_c_faithfulness_curves.csv"), index=False)

df_faithfulness = pd.DataFrame([{
    'Deletion_AUC_SHAP': round(float(deletion_auc_shap), 4),
    'Deletion_AUC_Random': round(float(deletion_auc_rand), 4),
    'Faithfulness_Deletion_Uplift': round(float(faithfulness_gap), 4),
    'Insertion_AUC_SHAP': round(float(insertion_auc_shap), 4),
    'Insertion_AUC_Random': round(float(insertion_auc_rand), 4),
    'Faithfulness_Insertion_Gain': round(float(insertion_auc_rand - insertion_auc_shap), 4)
}])
df_faithfulness.to_csv(os.path.join(OUT_DIR, "table_exp_c_faithfulness_auc.csv"), index=False)
print("\n[Exp C] Attribution Faithfulness AUC Summary:")
print(df_faithfulness)

print("\n[Exp C] Complete! Artifacts written to derived_matrices/")
