"""
Experiment D: Counterfactual Sparsity & Pareto Actionability Frontier
Evaluates:
- L0 Norm Sparsity: Distribution of Number of Active Intervention Levers (1, 2, 3Levers)
- Pareto Actionability Frontier: Cumulative Student Effort vs. Placement Probability Gain
- Physical Feasibility Compliance: Strict Bounded Feasibility Rate
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

print(f"[Exp D] Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df['Placement_Binary'] = (df['Placement'] == 'Yes').astype(int)
df['Internship_Binary'] = (df['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df[feature_cols].values
y = df['Placement_Binary'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[Exp D] Training production LightGBM classifier...")
model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=SEED, verbose=-1)
model.fit(X_scaled, y)

probs = model.predict_proba(X_scaled)[:, 1]

# Identify all unplaced at-risk students (prob < 0.30)
at_risk_idx = np.where((y == 0) & (probs < 0.30))[0]
print(f"[Exp D] Found {len(at_risk_idx)} at-risk students. Subsampling N=500 for Pareto evaluation...")

sample_idx = np.random.choice(at_risk_idx, size=min(500, len(at_risk_idx)), replace=False)

# Feature effort weights W (difficulty of modification):
# Internships (idx 4): weight 2.0 (high effort)
# Communication (idx 6): weight 1.0 (coaching/clinic)
# Projects (idx 7): weight 1.5 (capstone build)
# Academic features are held fixed as historical records (cost = inf)

tau_thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
pareto_records = []
sparsity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
total_solved = 0

# For default tau = 0.75
tau_default = 0.75
all_l0 = []
all_costs = []
all_gains = []

for idx in sample_idx:
    orig_x = X[idx].copy()
    orig_p = probs[idx]
    
    # Solve for multiple tau to trace the Pareto Frontier
    best_deltas_by_tau = {}
    
    for tau in tau_thresholds:
        best_cost = float('inf')
        best_delta = None
        best_p = orig_p
        
        # Grid search over feasible action space
        for d_comm in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            for d_proj in [0, 1, 2, 3]:
                for d_intern in [0, 1] if orig_x[4] == 0 else [0]:
                    if d_comm == 0 and d_proj == 0 and d_intern == 0:
                        continue
                    
                    # Compute weighted cost: L1 cost
                    cost = 1.0 * (d_comm / 4.0) + 1.5 * (d_proj / 3.0) + 2.0 * d_intern
                    if cost >= best_cost:
                        continue
                    
                    x_mod = orig_x.copy()
                    x_mod[6] = min(10.0, x_mod[6] + d_comm)
                    x_mod[7] = min(10, x_mod[7] + d_proj)
                    x_mod[4] = min(1, x_mod[4] + d_intern)
                    
                    x_mod_scaled = scaler.transform(x_mod.reshape(1, -1))
                    p_mod = model.predict_proba(x_mod_scaled)[0, 1]
                    
                    if p_mod >= tau:
                        best_cost = cost
                        best_delta = (d_comm, d_proj, d_intern)
                        best_p = p_mod
                        
        if best_delta is not None:
            best_deltas_by_tau[tau] = (best_cost, best_delta, best_p)
            
    # Record for default tau = 0.75
    if 0.80 in best_deltas_by_tau:
        cost, (dc, dp, di), p_mod = best_deltas_by_tau[0.80]
        l0 = int(dc > 0) + int(dp > 0) + int(di > 0)
        all_l0.append(l0)
        all_costs.append(cost)
        all_gains.append(p_mod - orig_p)
        sparsity_counts[l0] = sparsity_counts.get(l0, 0) + 1
        total_solved += 1

# 1. Sparsity Analysis
df_sparsity = pd.DataFrame([
    {'Active_Levers': '1 Feature Only', 'Student_Count': sparsity_counts[1], 'Percentage': round(sparsity_counts[1] / total_solved * 100, 2)},
    {'Active_Levers': '2 Features', 'Student_Count': sparsity_counts[2], 'Percentage': round(sparsity_counts[2] / total_solved * 100, 2)},
    {'Active_Levers': '3 Features', 'Student_Count': sparsity_counts[3], 'Percentage': round(sparsity_counts[3] / total_solved * 100, 2)},
])
mean_l0 = np.mean(all_l0)
print("\n[Exp D] Counterfactual Sparsity Distribution (tau = 0.80):")
print(df_sparsity)
print(f"Mean L0 Norm: {mean_l0:.2f} active intervention levers per student.")

df_sparsity.to_csv(os.path.join(OUT_DIR, "table_exp_d_counterfactual_sparsity.csv"), index=False)

# 2. Pareto Actionability Frontier
for tau in tau_thresholds:
    tau_costs = []
    tau_gains = []
    success_count = 0
    
    for idx in sample_idx:
        orig_p = probs[idx]
        orig_x = X[idx].copy()
        
        # Grid search
        best_cost = float('inf')
        best_p = orig_p
        for d_comm in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            for d_proj in [0, 1, 2, 3]:
                for d_intern in [0, 1] if orig_x[4] == 0 else [0]:
                    if d_comm == 0 and d_proj == 0 and d_intern == 0:
                        continue
                    cost = 1.0 * (d_comm / 4.0) + 1.5 * (d_proj / 3.0) + 2.0 * d_intern
                    if cost >= best_cost:
                        continue
                    x_mod = orig_x.copy()
                    x_mod[6] = min(10.0, x_mod[6] + d_comm)
                    x_mod[7] = min(10, x_mod[7] + d_proj)
                    x_mod[4] = min(1, x_mod[4] + d_intern)
                    x_mod_scaled = scaler.transform(x_mod.reshape(1, -1))
                    p_mod = model.predict_proba(x_mod_scaled)[0, 1]
                    if p_mod >= tau:
                        best_cost = cost
                        best_p = p_mod
                        
        if best_cost < float('inf'):
            tau_costs.append(best_cost)
            tau_gains.append(best_p - orig_p)
            success_count += 1
            
    pareto_records.append({
        'Target_Confidence_Tau': tau,
        'Feasible_Solution_Rate_Pct': round(success_count / len(sample_idx) * 100, 2),
        'Mean_Intervention_Effort': round(float(np.mean(tau_costs)), 4),
        'Std_Intervention_Effort': round(float(np.std(tau_costs)), 4),
        'Mean_Placement_Prob_Gain': round(float(np.mean(tau_gains)), 4),
        'Marginal_Effort_Per_10Pct_Gain': round(float(np.mean(tau_costs) / (np.mean(tau_gains) + 1e-6) * 0.10), 4)
    })

df_pareto = pd.DataFrame(pareto_records)
df_pareto.to_csv(os.path.join(OUT_DIR, "table_exp_d_pareto_actionability_frontier.csv"), index=False)
print("\n[Exp D] Pareto Actionability Frontier (Effort vs. Gain):")
print(df_pareto)

print("\n[Exp D] Complete! Artifacts written to derived_matrices/")
