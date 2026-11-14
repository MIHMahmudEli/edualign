"""
Module 4: Asymmetric Label Noise & Severe Imbalance Stress-Testing
Evaluates model robustness against asymmetric label flipping (social desirability bias)
and extreme class imbalance (1:1 to 1:20) comparing Standard Cross-Entropy, 
Cost-Sensitive Weighting, and Class-Balanced Focal Loss.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_recall_curve, auc, brier_score_loss
import lightgbm as lgb
from scipy.special import expit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("MODULE 4: ASYMMETRIC LABEL NOISE & SEVERE IMBALANCE STRESS-TESTING")
print("=" * 80)

# Load student career success dataset (N=50,000)
career_csv = os.path.join(RAW_DIR, 'kaggle', 'student-career-success-prediction-dataset', 'student_career_success_dataset.csv')
df = pd.read_csv(career_csv)

df['Leadership_Binary'] = df['Leadership_Experience'].map({'Yes': 1, 'No': 0}).fillna(0)
df['English_Score'] = df['English_Proficiency'].map({'Basic': 1, 'Intermediate': 2, 'Advanced': 3}).fillna(2)

feature_cols = ['CGPA', 'Study_Hours_Per_Week', 'Programming_Skill', 'Projects_Completed',
                'Certifications', 'Hackathons', 'Internships', 'Leadership_Binary',
                'Resume_Score', 'Communication_Skills', 'Teamwork', 'Problem_Solving',
                'English_Score', 'Interview_Score']

X_all = df[feature_cols].fillna(df[feature_cols].median()).values
y_all = (df['Placement_Status'] == 'Placed').astype(int).values

# Stratified clean test split (20% = 10,000 samples)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

scaler = StandardScaler()
X_train_full_s = scaler.fit_transform(X_train_full)
X_test_s = scaler.transform(X_test)

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# Define Focal Loss Objective for LightGBM scikit-learn API: objective(y_true, y_pred)
def focal_loss_objective(y_true, y_pred, gamma=2.0, alpha=0.25):
    labels = np.asarray(y_true)
    p = expit(y_pred)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    alpha_t = np.where(labels == 1, alpha, 1.0 - alpha)
    p_t = np.where(labels == 1, p, 1.0 - p)
    
    fl_weight = alpha_t * ((1.0 - p_t) ** gamma)
    grad = fl_weight * (p - labels)
    hess = fl_weight * p * (1.0 - p)
    hess = np.clip(hess, 1e-5, None)
    return grad, hess

# Imbalance regimes
# 1:1 (balanced, 50% pos)
# 1:3 (moderate, 25% pos)
# 1:10 (severe, ~9.1% pos)
# 1:20 (extreme, ~4.8% pos)
imbalance_ratios = {
    '1:1 (Balanced)': 1.0,
    '1:3 (Moderate)': 0.3333,
    '1:10 (Severe)': 0.10,
    '1:20 (Extreme)': 0.05
}

# Label noise levels: asymmetric flipping
# eta_0->1 = eta (unplaced falsely claimed placed)
# eta_1->0 = 0.5 * eta (placed falsely claimed unplaced)
noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

# Loss variants to benchmark
loss_methods = ['Standard Cross-Entropy', 'Cost-Sensitive Weighting', 'Class-Balanced Focal Loss']

all_results = []

pos_indices_full = np.where(y_train_full == 1)[0]
neg_indices_full = np.where(y_train_full == 0)[0]

np.random.seed(42)

for imb_name, imb_ratio in imbalance_ratios.items():
    print(f"\n--- Evaluating Imbalance Regime: {imb_name} ---")
    
    # Subsample training data to achieve exact ratio
    # Fixed number of negatives = 15,000
    n_neg = min(15000, len(neg_indices_full))
    n_pos = int(round(n_neg * imb_ratio))
    n_pos = min(n_pos, len(pos_indices_full))
    
    selected_neg = np.random.choice(neg_indices_full, size=n_neg, replace=False)
    selected_pos = np.random.choice(pos_indices_full, size=n_pos, replace=False)
    
    train_idx = np.concatenate([selected_neg, selected_pos])
    np.random.shuffle(train_idx)
    
    X_train_sub = X_train_full_s[train_idx]
    y_train_clean = y_train_full[train_idx]
    
    actual_pos_ratio = np.mean(y_train_clean)
    print(f"Cohort size: {len(y_train_clean)}, Positive Class Ratio: {actual_pos_ratio*100:.2f}%")
    
    for eta in noise_levels:
        # Asymmetric label corruption on training set only
        y_train_noisy = y_train_clean.copy()
        
        # 0 -> 1 flipping with prob eta
        mask_0 = (y_train_clean == 0)
        flip_0_to_1 = np.random.rand(np.sum(mask_0)) < eta
        y_train_noisy[np.where(mask_0)[0][flip_0_to_1]] = 1
        
        # 1 -> 0 flipping with prob 0.5 * eta
        mask_1 = (y_train_clean == 1)
        flip_1_to_0 = np.random.rand(np.sum(mask_1)) < (0.5 * eta)
        y_train_noisy[np.where(mask_1)[0][flip_1_to_0]] = 0
        
        for loss in loss_methods:
            if loss == 'Standard Cross-Entropy':
                clf = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.08,
                    random_state=42, verbose=-1
                )
                clf.fit(X_train_sub, y_train_noisy)
            elif loss == 'Cost-Sensitive Weighting':
                clf = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.08,
                    class_weight='balanced', random_state=42, verbose=-1
                )
                clf.fit(X_train_sub, y_train_noisy)
            elif loss == 'Class-Balanced Focal Loss':
                # Class balanced alpha
                pos_w = 1.0 - actual_pos_ratio
                neg_w = actual_pos_ratio
                clf = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.08,
                    objective=lambda y_t, y_p: focal_loss_objective(y_t, y_p, gamma=2.0, alpha=pos_w),
                    random_state=42, verbose=-1
                )
                clf.fit(X_train_sub, y_train_noisy)
            
            # Predict on pristine test set
            try:
                probs_raw = clf.predict_proba(X_test_s)
                if probs_raw.ndim == 2 and probs_raw.shape[1] == 2:
                    probs = probs_raw[:, 1]
                else:
                    probs = expit(clf.predict(X_test_s, raw_score=True))
            except Exception:
                probs = expit(clf.predict(X_test_s, raw_score=True))
            
            preds = (probs >= 0.50).astype(int)
            
            macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            minority_recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
            pr_auc = auc(recall_curve, precision_curve)
            ece = compute_ece(probs, y_test)
            brier = brier_score_loss(y_test, probs)
            
            all_results.append({
                'Imbalance_Regime': imb_name,
                'Noise_Level_Eta': f"{int(eta * 100)}%",
                'Loss_Function': loss,
                'Macro_F1': round(macro_f1, 4),
                'Minority_Recall': round(minority_recall, 4),
                'PR_AUC': round(pr_auc, 4),
                'ECE': round(ece, 4),
                'Brier_Score': round(brier, 4)
            })

df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(OUTPUT_DIR, 'table_exp_i_label_noise_imbalance.csv'), index=False)
print("\n[SUCCESS] Saved full results to table_exp_i_label_noise_imbalance.csv")

# Print summary comparison across noise levels for Extreme 1:20 and Moderate 1:3 regimes
print("\n=== SUMMARY AT EXTREME IMBALANCE (1:20) ===")
summary_ext = df_results[df_results['Imbalance_Regime'] == '1:20 (Extreme)']
print(summary_ext.pivot(index='Noise_Level_Eta', columns='Loss_Function', values='Macro_F1'))

print("\n=== SUMMARY AT MODERATE IMBALANCE (1:3) ===")
summary_mod = df_results[df_results['Imbalance_Regime'] == '1:3 (Moderate)']
print(summary_mod.pivot(index='Noise_Level_Eta', columns='Loss_Function', values='Macro_F1'))
