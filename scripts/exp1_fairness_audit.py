"""
Experiment 1: Algorithmic Fairness & Demographic Parity Audit
Evaluating Disparate Impact, Equal Opportunity, and Parity Differences across Gender and Regional Subgroups.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 1: ALGORITHMIC FAIRNESS & DEMOGRAPHIC PARITY AUDIT")
print("=" * 80)

def compute_fairness_metrics(y_true, y_pred, sensitive_attr, unprivileged_val, privileged_val):
    # Mask for subgroups
    mask_unpriv = (sensitive_attr == unprivileged_val)
    mask_priv = (sensitive_attr == privileged_val)
    
    # Selection Rates (Demographic Parity)
    sr_unpriv = np.mean(y_pred[mask_unpriv])
    sr_priv = np.mean(y_pred[mask_priv])
    dir_ratio = sr_unpriv / sr_priv if sr_priv > 0 else 1.0
    dpd = abs(sr_unpriv - sr_priv)
    
    # True Positive Rate (Equal Opportunity)
    tp_unpriv = np.sum((y_pred[mask_unpriv] == 1) & (y_true[mask_unpriv] == 1))
    p_unpriv = np.sum(y_true[mask_unpriv] == 1)
    tpr_unpriv = tp_unpriv / p_unpriv if p_unpriv > 0 else 0.0
    
    tp_priv = np.sum((y_pred[mask_priv] == 1) & (y_true[mask_priv] == 1))
    p_priv = np.sum(y_true[mask_priv] == 1)
    tpr_priv = tp_priv / p_priv if p_priv > 0 else 0.0
    eod = abs(tpr_unpriv - tpr_priv)
    
    # False Positive Rate
    fp_unpriv = np.sum((y_pred[mask_unpriv] == 1) & (y_true[mask_unpriv] == 0))
    n_unpriv = np.sum(y_true[mask_unpriv] == 0)
    fpr_unpriv = fp_unpriv / n_unpriv if n_unpriv > 0 else 0.0
    
    fp_priv = np.sum((y_pred[mask_priv] == 1) & (y_true[mask_priv] == 0))
    n_priv = np.sum(y_true[mask_priv] == 0)
    fpr_priv = fp_priv / n_priv if n_priv > 0 else 0.0
    
    aod = 0.5 * (abs(fpr_unpriv - fpr_priv) + abs(tpr_unpriv - tpr_priv))
    
    # Precision / Predictive Parity
    prec_unpriv = tp_unpriv / np.sum(y_pred[mask_unpriv] == 1) if np.sum(y_pred[mask_unpriv] == 1) > 0 else 0.0
    prec_priv = tp_priv / np.sum(y_pred[mask_priv] == 1) if np.sum(y_pred[mask_priv] == 1) > 0 else 0.0
    ppd = abs(prec_unpriv - prec_priv)
    
    return {
        'Selection_Rate_Unprivileged': round(float(sr_unpriv), 4),
        'Selection_Rate_Privileged': round(float(sr_priv), 4),
        'Disparate_Impact_Ratio': round(float(dir_ratio), 4),
        'Demographic_Parity_Diff': round(float(dpd), 4),
        'TPR_Unprivileged': round(float(tpr_unpriv), 4),
        'TPR_Privileged': round(float(tpr_priv), 4),
        'Equal_Opportunity_Diff': round(float(eod), 4),
        'Avg_Odds_Diff': round(float(aod), 4),
        'Predictive_Parity_Diff': round(float(ppd), 4),
        'EEOC_80_Rule_Satisfied': bool(0.80 <= dir_ratio <= 1.25)
    }

# -------------------------------------------------------------------------
# SUBGROUP AUDIT 1: Student Career Success (Gender: Female vs Male)
# -------------------------------------------------------------------------
print("\n[Audit 1] Student Career Success Cohort (N=50,000) - Gender Bias Audit...")
career_csv = os.path.join(RAW_DIR, 'kaggle', 'student-career-success-prediction-dataset', 'student_career_success_dataset.csv')
df_career = pd.read_csv(career_csv)

df_career['Leadership_Binary'] = df_career['Leadership_Experience'].map({'Yes': 1, 'No': 0}).fillna(0)
df_career['English_Score'] = df_career['English_Proficiency'].map({'Basic': 1, 'Intermediate': 2, 'Advanced': 3}).fillna(2)

career_features = ['CGPA', 'Study_Hours_Per_Week', 'Programming_Skill', 'Projects_Completed',
                   'Certifications', 'Hackathons', 'Internships', 'Leadership_Binary',
                   'Resume_Score', 'Communication_Skills', 'Teamwork', 'Problem_Solving',
                   'English_Score', 'Interview_Score']

y_career = (df_career['Placement_Status'] == 'Placed').astype(int).values
X_career = df_career[career_features].fillna(df_career[career_features].median()).values
gender_career = df_career['Gender'].values

# Out-of-fold predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_career = np.zeros_like(y_career)

for train_idx, test_idx in skf.split(X_career, y_career):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_career[train_idx])
    X_test_s = scaler.transform(X_career[test_idx])
    
    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    clf.fit(X_train_s, y_career[train_idx])
    y_pred_career[test_idx] = clf.predict(X_test_s)

res_gender_career = compute_fairness_metrics(y_career, y_pred_career, gender_career, 'Female', 'Male')
print("Career Success Gender Audit Results:")
for k, v in res_gender_career.items():
    print(f"  {k}: {v}")

# -------------------------------------------------------------------------
# SUBGROUP AUDIT 2: International Graduates (Gender: Female vs Male)
# -------------------------------------------------------------------------
print("\n[Audit 2] International Graduates Cohort (N=100,000 Sample) - Gender Bias Audit...")
intl_csv = os.path.join(RAW_DIR, 'kaggle', 'international-graduates-employment-dataset', 'dataset.csv')
df_intl = pd.read_csv(intl_csv, nrows=100000)

y_intl = (df_intl['Employment_Status'] == 'Employed').astype(int).values
df_intl['Internship_Binary'] = df_intl['Internship_Experience'].map({'Yes': 1, 'No': 0}).fillna(0)
df_intl['Ranking_Score'] = df_intl['University_Ranking'].map({'Low': 1, 'Medium': 2, 'High': 3}).fillna(2)
df_intl['Language_Score'] = df_intl['Language_Proficiency'].map({'Basic': 1, 'Intermediate': 2, 'Fluent': 3}).fillna(2)
df_intl['Education_Score'] = df_intl['Education_Level'].map({"Bachelor's": 1, "Master's": 2, "PhD": 3}).fillna(1)

intl_features = ['GPA', 'Age', 'Years_Since_Graduation', 'Internship_Binary', 'Ranking_Score', 'Language_Score', 'Education_Score']
X_intl = df_intl[intl_features].fillna(df_intl[intl_features].median()).values
gender_intl = df_intl['Gender'].values
region_intl = df_intl['Region_of_Study'].values

y_pred_intl = np.zeros_like(y_intl)
for train_idx, test_idx in skf.split(X_intl, y_intl):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_intl[train_idx])
    X_test_s = scaler.transform(X_intl[test_idx])
    
    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    clf.fit(X_train_s, y_intl[train_idx])
    y_pred_intl[test_idx] = clf.predict(X_test_s)

res_gender_intl = compute_fairness_metrics(y_intl, y_pred_intl, gender_intl, 'Female', 'Male')
print("\nInternational Graduates Gender Audit Results:")
for k, v in res_gender_intl.items():
    print(f"  {k}: {v}")

# -------------------------------------------------------------------------
# SUBGROUP AUDIT 3: International Graduates (Region: Non-US/UK vs US/UK)
# -------------------------------------------------------------------------
print("\n[Audit 3] International Graduates Cohort - Regional Parity Audit...")
top_regions = df_intl['Region_of_Study'].value_counts()
print(f"Region counts:\n{top_regions}")

# Compare EU vs US/UK or top 2 regions
reg1, reg2 = top_regions.index[0], top_regions.index[1]
print(f"Comparing region {reg1} vs {reg2}...")
res_region_intl = compute_fairness_metrics(y_intl, y_pred_intl, region_intl, reg1, reg2)
print(f"\nRegional Audit ({reg1} vs {reg2}):")
for k, v in res_region_intl.items():
    print(f"  {k}: {v}")

# Compile into summary table
fairness_rows = [
    {'Evaluation_Setting': 'Student Career Success (Gender: Female vs Male)', **res_gender_career},
    {'Evaluation_Setting': 'International Graduates (Gender: Female vs Male)', **res_gender_intl},
    {'Evaluation_Setting': f'International Graduates (Region: {reg1} vs {reg2})', **res_region_intl}
]
df_fairness = pd.DataFrame(fairness_rows)
df_fairness.to_csv(os.path.join(OUTPUT_DIR, 'table_exp1_algorithmic_fairness_audit.csv'), index=False)
print("\nSaved: table_exp1_algorithmic_fairness_audit.csv")
print(df_fairness[['Evaluation_Setting', 'Disparate_Impact_Ratio', 'Equal_Opportunity_Diff', 'EEOC_80_Rule_Satisfied']])

print("\nEXPERIMENT 1 COMPLETED!")
