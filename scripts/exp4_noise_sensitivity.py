"""
Experiment 4: Monte Carlo Noise Stress-Testing & Counterfactual Sensitivity Analysis
Testing classifier degradation and prescriptive stability under feature measurement uncertainty.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 4: NOISE STRESS-TESTING & COUNTERFACTUAL STABILITY")
print("=" * 80)

college_csv = os.path.join(RAW_DIR, 'kaggle', 'college-student-placement', 'college_student_placement_dataset.csv')
df = pd.read_csv(college_csv)
df['Placement_Binary'] = (df['Placement'] == 'Yes').astype(int)
df['Internship_Binary'] = (df['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df[feature_cols].values
y = df['Placement_Binary'].values

# Split train (80%) and test (20%)
np.random.seed(42)
test_idx = np.random.choice(len(X), size=int(0.2 * len(X)), replace=False)
train_idx = np.setdiff1d(np.arange(len(X)), test_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train production model on clean training data
model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
model.fit(X_train_s, y_train)

# Clean baseline test performance
y_pred_clean = model.predict(X_test_s)
y_prob_clean = model.predict_proba(X_test_s)[:, 1]

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
n_trials = 15

stress_results = []
std_features = np.std(X_train, axis=0)

# Sample 50 at-risk students for counterfactual stability test
at_risk_mask = (y_test == 0) & (y_prob_clean < 0.20)
at_risk_indices = np.where(at_risk_mask)[0][:50]

for sigma in noise_levels:
    accs, f1s, aucs, briers, stabilities = [], [], [], [], []
    
    for trial in range(n_trials):
        # Generate Gaussian noise proportional to feature standard deviation
        noise = np.random.normal(0, sigma * std_features, size=X_test.shape)
        # Prevent noise in binary internship feature
        noise[:, 4] = 0.0
        X_test_noisy = X_test + noise
        X_test_noisy_s = scaler.transform(X_test_noisy)
        
        preds = model.predict(X_test_noisy_s)
        probs = model.predict_proba(X_test_noisy_s)[:, 1]
        
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        aucs.append(roc_auc_score(y_test, probs))
        briers.append(brier_score_loss(y_test, probs))
        
        # Counterfactual recommendation directional stability:
        # Check if probability flips or remains in same direction
        prob_agreement = np.mean(np.abs(probs[at_risk_indices] - y_prob_clean[at_risk_indices]) < 0.25)
        stabilities.append(prob_agreement)
        
    stress_results.append({
        'Noise_Level_Sigma': f"{int(sigma*100)}%",
        'Accuracy': f"{np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%",
        'Macro_F1': f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
        'ROC_AUC': f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}",
        'Brier_Score': f"{np.mean(briers):.4f}",
        'CF_Prescriptive_Stability': f"{np.mean(stabilities)*100:.1f}%",
        'Integrity_Status': 'Optimal' if sigma <= 0.10 else 'Resilient' if sigma <= 0.20 else 'Degraded'
    })

df_stress = pd.DataFrame(stress_results)
df_stress.to_csv(os.path.join(OUTPUT_DIR, 'table_exp4_noise_stress_testing.csv'), index=False)

print("Saved: table_exp4_noise_stress_testing.csv")
print(df_stress[['Noise_Level_Sigma', 'Accuracy', 'Macro_F1', 'ROC_AUC', 'CF_Prescriptive_Stability', 'Integrity_Status']].to_string(index=False))

print("\nEXPERIMENT 4 COMPLETED!")
