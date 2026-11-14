"""
Generate exact ROC curves and Confusion Matrices across all evaluated models
for the Domestic College Placement benchmark (N=10,000).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING ROC CURVES AND CONFUSION MATRICES ACROSS ALL MODELS")
print("=" * 80)

# Load dataset
csv_path = os.path.join(RAW_DIR, 'kaggle', 'college-student-placement', 'college_student_placement_dataset.csv')
df = pd.read_csv(csv_path)

df['Placement_Binary'] = (df['Placement'] == 'Yes').astype(int)
df['Internship_Binary'] = (df['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df[feature_cols].values
y = df['Placement_Binary'].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Define models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, verbose=-1),
    'Deep Tabular MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
}

# Stacking Meta-Learner
estimators = [
    ('dt', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1))
]
models['Stacking Meta-Ensemble'] = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3
)

# Perform out-of-fold predictions using 5-Fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = {m: np.zeros(len(y)) for m in models}
oof_preds = {m: np.zeros(len(y), dtype=int) for m in models}

print("Running 5-fold cross-validation for out-of-fold ROC and confusion matrix derivation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_s, y)):
    X_train, X_val = X_s[train_idx], X_s[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[name][val_idx] = probs
        oof_preds[name][val_idx] = (probs >= 0.50).astype(int)

# 1. Compute ROC curves and subsample points for TikZ plotting
roc_data = []
print("\n--- Model ROC AUC Results ---")
for name in models:
    fpr, tpr, thresholds = roc_curve(y, oof_probs[name])
    roc_auc = auc(fpr, tpr)
    print(f"{name:25s} ROC-AUC: {roc_auc:.4f}")
    
    # Subsample 15-20 representative points for clean PGFPlots rendering
    sample_indices = np.linspace(0, len(fpr) - 1, 25, dtype=int)
    # Always include boundary points (0,0) and (1,1)
    sample_indices = np.unique(np.concatenate([[0], sample_indices, [len(fpr)-1]]))
    
    for idx in sample_indices:
        roc_data.append({
            'Model': name,
            'FPR': round(float(fpr[idx]), 4),
            'TPR': round(float(tpr[idx]), 4),
            'AUC': round(float(roc_auc), 4)
        })

df_roc = pd.DataFrame(roc_data)
df_roc.to_csv(os.path.join(OUTPUT_DIR, 'table_model_roc_curves.csv'), index=False)
print("Saved table_model_roc_curves.csv")

# 2. Compute Confusion Matrices for all models
cm_data = []
print("\n--- Confusion Matrices (N=10,000) ---")
for name in models:
    cm = confusion_matrix(y, oof_preds[name])
    tn, fp, fn, tp = cm.ravel()
    n_total = len(y)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n{name}:")
    print(f"  TN: {tn:5d} ({tn/n_total*100:5.2f}%)  |  FP: {fp:5d} ({fp/n_total*100:5.2f}%)")
    print(f"  FN: {fn:5d} ({fn/n_total*100:5.2f}%)  |  TP: {tp:5d} ({tp/n_total*100:5.2f}%)")
    print(f"  Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, Specificity: {spec*100:.2f}%")
    
    cm_data.append({
        'Model': name,
        'TN': tn, 'TN_pct': round(tn/n_total*100, 2),
        'FP': fp, 'FP_pct': round(fp/n_total*100, 2),
        'FN': fn, 'FN_pct': round(fn/n_total*100, 2),
        'TP': tp, 'TP_pct': round(tp/n_total*100, 2),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'Specificity': round(spec, 4),
        'NPV': round(npv, 4)
    })

df_cm = pd.DataFrame(cm_data)
df_cm.to_csv(os.path.join(OUTPUT_DIR, 'table_confusion_matrices.csv'), index=False)
print("\nSaved table_confusion_matrices.csv")
