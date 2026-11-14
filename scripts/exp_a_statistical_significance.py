"""
Experiment A: Rigorous Statistical Significance & Hypothesis Testing Benchmark
Optimized for high-speed deterministic single-pass execution.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
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

print(f"[Exp A] Loading dataset: {DATA_PATH}", flush=True)
df_college = pd.read_csv(DATA_PATH)
df_college['Placement_Binary'] = (df_college['Placement'] == 'Yes').astype(int)
df_college['Internship_Binary'] = (df_college['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df_college[feature_cols].values
y = df_college['Placement_Binary'].values
N, D = X.shape
print(f"[Exp A] N={N}, D={D}, Positive Placed={y.sum()} ({y.mean():.2%})", flush=True)

# Define 7 fast, robust models with n_jobs=1 to avoid Windows loky overhead
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=6, ccp_alpha=0.001, random_state=SEED),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=500, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED, n_jobs=1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=50, random_state=SEED),
    'LightGBM': lgb.LGBMClassifier(n_estimators=50, max_depth=6, random_state=SEED, n_jobs=1, verbose=-1),
    'Deep Tabular MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, early_stopping=True, random_state=SEED),
    'Stacking Meta-Ensemble': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=30, max_depth=8, random_state=SEED, n_jobs=1)),
            ('hgb', HistGradientBoostingClassifier(max_iter=30, random_state=SEED)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=SEED))
        ],
        final_estimator=LogisticRegression(class_weight='balanced', random_state=SEED),
        cv=3,
        n_jobs=1
    )
}

# 10-Fold Stratified Cross-Validation (SINGLE PASS)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

fold_f1 = {m: [] for m in models}
fold_auc = {m: [] for m in models}
fold_bacc = {m: [] for m in models}
fold_brier = {m: [] for m in models}

oof_preds = {m: np.zeros(N) for m in models}
oof_probs = {m: np.zeros(N) for m in models}

print("[Exp A] Executing single-pass 10-fold Stratified CV across 7 models...", flush=True)
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, clf in models.items():
        if name in ['Logistic Regression', 'Deep Tabular MLP', 'Stacking Meta-Ensemble']:
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_test_scaled)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            
        oof_preds[name][test_idx] = preds
        oof_probs[name][test_idx] = probs
        
        f1 = f1_score(y_test, preds, average='macro')
        auc = roc_auc_score(y_test, probs)
        bacc = balanced_accuracy_score(y_test, preds)
        brier = brier_score_loss(y_test, probs)
        
        fold_f1[name].append(f1)
        fold_auc[name].append(auc)
        fold_bacc[name].append(bacc)
        fold_brier[name].append(brier)
    
    print(f"  --> Fold {fold_idx+1}/10 completed successfully.", flush=True)

# 1. Pairwise Wilcoxon Signed-Rank Test vs Stacking Meta-Ensemble
meta_name = 'Stacking Meta-Ensemble'
wilcoxon_results = []

for name in models:
    if name == meta_name:
        continue
    
    diff_f1 = np.array(fold_f1[meta_name]) - np.array(fold_f1[name])
    diff_auc = np.array(fold_auc[meta_name]) - np.array(fold_auc[name])
    
    if np.all(diff_f1 == 0):
        w_stat_f1, p_val_f1 = 0.0, 1.0
    else:
        try:
            w_stat_f1, p_val_f1 = stats.wilcoxon(fold_f1[meta_name], fold_f1[name], zero_method='wilcox')
        except Exception:
            w_stat_f1, p_val_f1 = 0.0, 1.0
            
    if np.all(diff_auc == 0):
        w_stat_auc, p_val_auc = 0.0, 1.0
    else:
        try:
            w_stat_auc, p_val_auc = stats.wilcoxon(fold_auc[meta_name], fold_auc[name], zero_method='wilcox')
        except Exception:
            w_stat_auc, p_val_auc = 0.0, 1.0
            
    # Cohen's d (Macro F1)
    pooled_std = np.sqrt((np.var(fold_f1[meta_name], ddof=1) + np.var(fold_f1[name], ddof=1)) / 2.0)
    cohen_d = (np.mean(fold_f1[meta_name]) - np.mean(fold_f1[name])) / (pooled_std + 1e-12)
    
    # Paired t-test
    if np.all(diff_f1 == 0):
        t_stat, p_val_t = 0.0, 1.0
    else:
        t_stat, p_val_t = stats.ttest_rel(fold_f1[meta_name], fold_f1[name])
    
    sig_label = 'p < 0.001' if p_val_f1 < 0.001 else ('p < 0.01' if p_val_f1 < 0.01 else ('p < 0.05' if p_val_f1 < 0.05 else 'n.s.'))
    
    wilcoxon_results.append({
        'Baseline_Model': name,
        'Meta_Macro_F1': round(float(np.mean(fold_f1[meta_name])), 4),
        'Baseline_Macro_F1': round(float(np.mean(fold_f1[name])), 4),
        'Delta_F1': round(float(np.mean(fold_f1[meta_name]) - np.mean(fold_f1[name])), 4),
        'Wilcoxon_W': round(float(w_stat_f1), 2),
        'Wilcoxon_p_F1': float(p_val_f1),
        'Wilcoxon_p_AUC': float(p_val_auc),
        'Paired_t_stat': round(float(t_stat), 3),
        'Paired_t_p': float(p_val_t),
        'Cohens_d': round(float(cohen_d), 3),
        'Significance': sig_label
    })

df_wilcoxon = pd.DataFrame(wilcoxon_results)
df_wilcoxon.to_csv(os.path.join(OUT_DIR, "table_exp_a_statistical_tests.csv"), index=False)
print("\n[Exp A] Pairwise Wilcoxon Signed-Rank Test Results:", flush=True)
print(df_wilcoxon[['Baseline_Model', 'Delta_F1', 'Wilcoxon_p_F1', 'Cohens_d', 'Significance']], flush=True)

# 2. Friedman Omnibus Test across all 7 models
matrix_f1 = np.array([fold_f1[m] for m in models]).T
friedman_stat, friedman_p = stats.friedmanchisquare(*[fold_f1[m] for m in models])

N_folds = 10
k_models = len(models)
chi_F = friedman_stat
F_F = ((N_folds - 1) * chi_F) / (N_folds * (k_models - 1) - chi_F + 1e-12)
p_iman_davenport = stats.f.sf(F_F, k_models - 1, (k_models - 1) * (N_folds - 1))

print(f"\n[Exp A] Friedman chi^2 = {friedman_stat:.4f}, p = {friedman_p:.4e}", flush=True)
print(f"[Exp A] Iman-Davenport F_F = {F_F:.4f}, p = {p_iman_davenport:.4e}", flush=True)

# Compute Average Ranks
ranks = np.zeros_like(matrix_f1)
for i in range(N_folds):
    ranks[i] = stats.rankdata(-matrix_f1[i])

avg_ranks = np.mean(ranks, axis=0)
model_names_list = list(models.keys())
df_ranks = pd.DataFrame({
    'Model': model_names_list,
    'Average_Rank': avg_ranks
}).sort_values('Average_Rank')

q_05 = 2.949
cd_05 = q_05 * np.sqrt((k_models * (k_models + 1)) / (6 * N_folds))
print(f"[Exp A] Nemenyi Critical Difference CD(alpha=0.05) = {cd_05:.4f}", flush=True)

# 3. 1,000-sample Empirical Bootstrap Confidence Intervals (Fast Vectorized)
print("\n[Exp A] Computing 1,000-sample Empirical Bootstrap 95% Confidence Intervals...", flush=True)
B = 1000
bootstrap_records = []

# Generate all 1,000 sample indices upfront
boot_indices = np.random.randint(0, N, size=(B, N))

for name in models:
    preds = oof_preds[name]
    probs = oof_probs[name]
    
    boot_f1 = np.zeros(B)
    boot_auc = np.zeros(B)
    boot_bacc = np.zeros(B)
    
    for b in range(B):
        idx = boot_indices[b]
        y_b = y[idx]
        pred_b = preds[idx]
        prob_b = probs[idx]
        
        boot_f1[b] = f1_score(y_b, pred_b, average='macro')
        boot_auc[b] = roc_auc_score(y_b, prob_b)
        boot_bacc[b] = balanced_accuracy_score(y_b, pred_b)
        
    f1_low, f1_high = np.percentile(boot_f1, [2.5, 97.5])
    auc_low, auc_high = np.percentile(boot_auc, [2.5, 97.5])
    bacc_low, bacc_high = np.percentile(boot_bacc, [2.5, 97.5])
    
    bootstrap_records.append({
        'Model': name,
        'Macro_F1_mean': round(float(np.mean(boot_f1)), 4),
        'Macro_F1_CI_95': f"[{f1_low:.4f}, {f1_high:.4f}]",
        'ROC_AUC_mean': round(float(np.mean(boot_auc)), 4),
        'ROC_AUC_CI_95': f"[{auc_low:.4f}, {auc_high:.4f}]",
        'Balanced_Acc_mean': round(float(np.mean(boot_bacc)), 4),
        'Balanced_Acc_CI_95': f"[{bacc_low:.4f}, {bacc_high:.4f}]"
    })

df_boot = pd.DataFrame(bootstrap_records)
df_boot.to_csv(os.path.join(OUT_DIR, "table_exp_a_bootstrap_ci.csv"), index=False)
print("\n[Exp A] 1,000-Iteration Bootstrap 95% Confidence Intervals:", flush=True)
print(df_boot[['Model', 'Macro_F1_CI_95', 'ROC_AUC_CI_95']], flush=True)

summary_txt = f"""Statistical Significance Benchmark Summary:
Friedman chi^2 = {friedman_stat:.4f} (p = {friedman_p:.4e})
Iman-Davenport F_F = {F_F:.4f} (p = {p_iman_davenport:.4e})
Nemenyi Critical Difference CD(alpha=0.05) = {cd_05:.4f}

Average Ranks:
{df_ranks.to_string(index=False)}
"""
with open(os.path.join(OUT_DIR, "exp_a_statistical_summary.txt"), "w") as f:
    f.write(summary_txt)

print("\n[Exp A] Complete! Artifacts written to derived_matrices/", flush=True)
