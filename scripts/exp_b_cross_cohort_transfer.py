"""
Experiment B: Cross-Cohort Zero-Shot Transfer & Covariate Shift Analysis
Evaluates:
- Unified 3-Dimensional Core Competency Vector: [Academic, Communication, Practical Experience]
- Zero-Shot Transfer Matrix across 3 Cohorts (Domestic 10K, Career Success 50K, International 50K)
- Distributional Covariate Shift: Wasserstein Distance (W1), Jensen-Shannon Divergence (D_JS), MMD
- Few-Shot Target Calibration Recovery Curve (0%, 5%, 10%, 20%)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, brier_score_loss
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "derived_matrices")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Cohort 1: Domestic College Placement (N=10,000)
c1_path = os.path.join(RAW_DIR, "kaggle", "college-student-placement", "college_student_placement_dataset.csv")
df1 = pd.read_csv(c1_path)
df1['Placement_Binary'] = (df1['Placement'] == 'Yes').astype(int)
df1['Internship_Binary'] = (df1['Internship_Experience'] == 'Yes').astype(int)

# Normalize Cohort 1 to [0, 1] unified space
gpa_c1 = (df1['CGPA'] - df1['CGPA'].min()) / (df1['CGPA'].max() - df1['CGPA'].min())
comm_c1 = (df1['Communication_Skills'] - df1['Communication_Skills'].min()) / (df1['Communication_Skills'].max() - df1['Communication_Skills'].min())
pract_raw_c1 = df1['Projects_Completed'] + 1.5 * df1['Internship_Binary']
pract_c1 = (pract_raw_c1 - pract_raw_c1.min()) / (pract_raw_c1.max() - pract_raw_c1.min())

X1 = np.column_stack([gpa_c1, comm_c1, pract_c1])
y1 = df1['Placement_Binary'].values

# 2. Load Cohort 2: Student Career Success (N=50,000)
c2_path = os.path.join(RAW_DIR, "kaggle", "student-career-success-prediction-dataset", "student_career_success_dataset.csv")
df2 = pd.read_csv(c2_path)
df2['Placement_Binary'] = (df2['Placement_Status'] == 'Placed').astype(int)

gpa_c2 = (df2['CGPA'] - df2['CGPA'].min()) / (df2['CGPA'].max() - df2['CGPA'].min())
comm_c2 = (df2['Communication_Skills'] - df2['Communication_Skills'].min()) / (df2['Communication_Skills'].max() - df2['Communication_Skills'].min())
pract_raw_c2 = df2['Projects_Completed'] + 1.5 * df2['Internships'].fillna(0)
pract_c2 = (pract_raw_c2 - pract_raw_c2.min()) / (pract_raw_c2.max() - pract_raw_c2.min())

X2 = np.column_stack([gpa_c2, comm_c2, pract_c2])
y2 = df2['Placement_Binary'].values

# 3. Load Cohort 3: International Graduates (Sample N=50,000 from 300,000 for efficiency)
c3_path = os.path.join(RAW_DIR, "kaggle", "international-graduates-employment-dataset", "dataset.csv")
df3 = pd.read_csv(c3_path, nrows=100000)
# Subsample 50,000
df3 = df3.sample(n=50000, random_state=SEED)
df3['Employment_Binary'] = (df3['Employment_Status'] == 'Employed').astype(int)
df3['Internship_Binary'] = (df3['Internship_Experience'] == 'Yes').astype(int)
lang_map = {'Basic': 0.25, 'Intermediate': 0.50, 'Advanced': 0.75, 'Fluent': 1.0}
comm_c3 = df3['Language_Proficiency'].map(lang_map).fillna(0.5).values

gpa_c3 = (df3['GPA'] - df3['GPA'].min()) / (df3['GPA'].max() - df3['GPA'].min())
pract_raw_c3 = df3['Internship_Binary'].values.astype(float)
pract_c3 = pract_raw_c3

X3 = np.column_stack([gpa_c3, comm_c3, pract_c3])
y3 = df3['Employment_Binary'].values

cohorts = {
    'Domestic (N=10K)': (X1, y1),
    'Career Success (N=50K)': (X2, y2),
    'International (N=50K)': (X3, y3)
}

print("[Exp B] Loaded 3 cohorts in unified space [Academic, Communication, Practical]:")
for c_name, (Xc, yc) in cohorts.items():
    print(f"  {c_name}: shape={Xc.shape}, positive_rate={yc.mean():.2%}")

# Split each cohort 80/20 train/test
splits = {}
for c_name, (Xc, yc) in cohorts.items():
    X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.20, random_state=SEED, stratify=yc)
    splits[c_name] = {'train': (X_tr, y_tr), 'test': (X_te, y_te)}

# Train LightGBM on each source cohort
trained_models = {}
for c_name, data in splits.items():
    X_tr, y_tr = data['train']
    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=SEED, verbose=-1)
    clf.fit(X_tr, y_tr)
    trained_models[c_name] = clf

# 1. Zero-Shot Cross-Cohort Transfer Matrix
print("\n[Exp B] Computing Zero-Shot Cross-Cohort Transfer Matrix...")
transfer_records = []
cohort_names = list(cohorts.keys())

# In-domain baseline AUCs
in_domain_auc = {}
for c_target in cohort_names:
    X_te, y_te = splits[c_target]['test']
    probs_in = trained_models[c_target].predict_proba(X_te)[:, 1]
    in_domain_auc[c_target] = roc_auc_score(y_te, probs_in)

for s_name in cohort_names:
    for t_name in cohort_names:
        X_te, y_te = splits[t_name]['test']
        clf = trained_models[s_name]
        probs = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)
        
        auc = roc_auc_score(y_te, probs)
        f1 = f1_score(y_te, preds, average='macro')
        acc = accuracy_score(y_te, preds)
        
        is_in_domain = (s_name == t_name)
        target_in_auc = in_domain_auc[t_name]
        trr = (auc / target_in_auc) * 100.0 # Transfer Retention Ratio (%)
        deg_gap = target_in_auc - auc        # Degradation Gap
        
        transfer_records.append({
            'Source_Cohort': s_name,
            'Target_Cohort': t_name,
            'Is_In_Domain': is_in_domain,
            'ROC_AUC': round(float(auc), 4),
            'Macro_F1': round(float(f1), 4),
            'Accuracy': round(float(acc), 4),
            'Transfer_Retention_Rate_Pct': round(float(trr), 2),
            'Degradation_Gap_AUC': round(float(deg_gap), 4)
        })

df_transfer = pd.DataFrame(transfer_records)
df_transfer.to_csv(os.path.join(OUT_DIR, "table_exp_b_zero_shot_transfer_matrix.csv"), index=False)
print(df_transfer[['Source_Cohort', 'Target_Cohort', 'ROC_AUC', 'Transfer_Retention_Rate_Pct', 'Degradation_Gap_AUC']])

# 2. Covariate Shift Quantification (Wasserstein Distance & MMD)
print("\n[Exp B] Computing Covariate Shift Quantification (Wasserstein W1, Jensen-Shannon, MMD)...")

def compute_mmd_rbf(X, Y, gamma=1.0, max_n=2000):
    # Subsample for fast MMD estimation
    if len(X) > max_n:
        X = X[np.random.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[np.random.choice(len(Y), max_n, replace=False)]
    
    K_XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    K_YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    K_XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    
    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return np.sqrt(max(0.0, mmd2))

covariate_records = []
pairs = [
    ('Domestic (N=10K)', 'Career Success (N=50K)'),
    ('Domestic (N=10K)', 'International (N=50K)'),
    ('Career Success (N=50K)', 'International (N=50K)')
]

feature_names = ['Academic_Capital', 'Communication_Skills', 'Practical_Experience']

for cA, cB in pairs:
    XA = cohorts[cA][0]
    XB = cohorts[cB][0]
    
    # 1D Wasserstein distances
    w1_academic = stats.wasserstein_distance(XA[:, 0], XB[:, 0])
    w1_comm = stats.wasserstein_distance(XA[:, 1], XB[:, 1])
    w1_pract = stats.wasserstein_distance(XA[:, 2], XB[:, 2])
    mean_w1 = np.mean([w1_academic, w1_comm, w1_pract])
    
    # Jensen-Shannon Divergence across joint feature marginals
    js_divergences = []
    for f_i in range(3):
        histA, edges = np.histogram(XA[:, f_i], bins=20, range=(0, 1), density=True)
        histB, _ = np.histogram(XB[:, f_i], bins=edges, density=True)
        # Add epsilon and normalize
        p = histA + 1e-10
        p /= p.sum()
        q = histB + 1e-10
        q /= q.sum()
        js_divergences.append(jensenshannon(p, q))
    mean_js = np.mean(js_divergences)
    
    # Multidimensional MMD
    mmd_val = compute_mmd_rbf(XA, XB, gamma=1.0)
    
    covariate_records.append({
        'Cohort_Pair': f"{cA} vs. {cB}",
        'W1_Academic': round(float(w1_academic), 4),
        'W1_Communication': round(float(w1_comm), 4),
        'W1_Practical': round(float(w1_pract), 4),
        'Mean_Wasserstein_W1': round(float(mean_w1), 4),
        'Jensen_Shannon_Div': round(float(mean_js), 4),
        'RBF_MMD': round(float(mmd_val), 4)
    })

df_covariate = pd.DataFrame(covariate_records)
df_covariate.to_csv(os.path.join(OUT_DIR, "table_exp_b_covariate_shift_metrics.csv"), index=False)
print("\n[Exp B] Covariate Shift Metrics:")
print(df_covariate)

# 3. Few-Shot Target Calibration Recovery Curve
print("\n[Exp B] Computing Few-Shot Domain Calibration Recovery Curve...")
# Source: Domestic (N=10K), Target: Career Success (N=50K) and International (N=50K)
adaptation_records = []
fractions = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0]

for t_name in ['Career Success (N=50K)', 'International (N=50K)']:
    X_tr_t, y_tr_t = splits[t_name]['train']
    X_te_t, y_te_t = splits[t_name]['test']
    
    base_src_model = trained_models['Domestic (N=10K)']
    
    for frac in fractions:
        if frac == 0.0:
            # Pure zero-shot
            probs = base_src_model.predict_proba(X_te_t)[:, 1]
        elif frac == 1.0:
            # Full target training
            target_model = trained_models[t_name]
            probs = target_model.predict_proba(X_te_t)[:, 1]
        else:
            # Few-shot fine-tune / combined training
            n_calib = int(len(X_tr_t) * frac)
            calib_idx = np.random.choice(len(X_tr_t), size=n_calib, replace=False)
            
            # Combine source train + target calib with sample weighting
            X_combined = np.vstack([splits['Domestic (N=10K)']['train'][0], X_tr_t[calib_idx]])
            y_combined = np.hstack([splits['Domestic (N=10K)']['train'][1], y_tr_t[calib_idx]])
            weights = np.hstack([
                np.ones(len(splits['Domestic (N=10K)']['train'][0])),
                np.ones(n_calib) * 3.0 # Upweight target calibration samples
            ])
            
            adapted_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=SEED, verbose=-1)
            adapted_model.fit(X_combined, y_combined, sample_weight=weights)
            probs = adapted_model.predict_proba(X_te_t)[:, 1]
            
        auc = roc_auc_score(y_te_t, probs)
        retention = (auc / in_domain_auc[t_name]) * 100.0
        
        adaptation_records.append({
            'Target_Cohort': t_name,
            'Calibration_Fraction': f"{int(frac*100)}%",
            'ROC_AUC': round(float(auc), 4),
            'AUC_Recovery_Pct': round(float(retention), 2)
        })

df_adapt = pd.DataFrame(adaptation_records)
df_adapt.to_csv(os.path.join(OUT_DIR, "table_exp_b_few_shot_adaptation_curve.csv"), index=False)
print("\n[Exp B] Few-Shot Domain Calibration Recovery:")
print(df_adapt)

print("\n[Exp B] Execution successfully completed! Artifacts saved in derived_matrices/.")
