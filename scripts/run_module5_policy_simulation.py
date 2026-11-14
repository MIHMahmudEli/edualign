"""
Module 5: Macro-Cohort Institutional Impact & Longitudinal Policy Simulation
Simulates university-wide rollout across N=10,000 students under varying 
compliance rates alpha in [0%, 100%]. Evaluates cohort placement yield, 
additional graduates placed, equity gap reduction, and institutional ROI.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("MODULE 5: MACRO-COHORT INSTITUTIONAL IMPACT & POLICY SIMULATION")
print("=" * 80)

# Load student dataset (N=10,000 college placement dataset)
college_csv = os.path.join(RAW_DIR, 'kaggle', 'college-student-placement', 'college_student_placement_dataset.csv')
df = pd.read_csv(college_csv)

df['Placement_Binary'] = (df['Placement'] == 'Yes').astype(int)
df['Internship_Binary'] = (df['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df[feature_cols].values
y = df['Placement_Binary'].values

# Fit production model on entire cohort (or standard train split)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
model.fit(X_s, y)

# Clean baseline predicted placement probability
probs_baseline = model.predict_proba(X_s)[:, 1]
preds_baseline = (probs_baseline >= 0.50).astype(int)

# Identify at-risk sub-cohort (unplaced under status quo)
at_risk_mask = (preds_baseline == 0)
n_at_risk = int(np.sum(at_risk_mask))
n_total = len(df)
base_yield = np.mean(preds_baseline)

print(f"Total Cohort Size: {n_total}")
print(f"Status Quo Placed Students: {int(np.sum(preds_baseline))} ({base_yield*100:.2f}%)")
print(f"Status Quo At-Risk (Unplaced) Students: {n_at_risk} ({(1-base_yield)*100:.2f}%)")

# Cost and financial parameters for Institutional ROI:
# Cost per student intervention: $350 (mentorship, cloud lab credits, resume/interview workshops)
# Economic value per placed graduate: $8,500 (tuition premium, alumni contribution, institutional ranking subsidy)
COST_PER_STUDENT = 350.0
VALUE_PER_PLACEMENT = 8500.0

compliance_rates = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
n_simulations = 50

simulation_results = []
np.random.seed(42)

for alpha in compliance_rates:
    yields = []
    addl_placed_list = []
    net_benefits = []
    cbr_list = []
    
    for sim in range(n_simulations):
        X_sim = X.copy()
        
        # For each at-risk student, compliance occurs with probability alpha
        at_risk_indices = np.where(at_risk_mask)[0]
        compliant_mask = np.random.rand(len(at_risk_indices)) < alpha
        treated_indices = at_risk_indices[compliant_mask]
        
        # Apply prescriptive intervention to treated students:
        # 1. Projects Completed: +1 to +2 projects (causal feasibility <= 45h - 90h)
        # 2. Communication Skills: +10% to +20% score uplift
        if len(treated_indices) > 0:
            # Feature index 7: Projects_Completed
            proj_bump = np.random.choice([1, 2], size=len(treated_indices), p=[0.4, 0.6])
            X_sim[treated_indices, 7] = np.clip(X_sim[treated_indices, 7] + proj_bump, 0, 10)
            
            # Feature index 6: Communication_Skills (scale 0-100 or standardized)
            comm_bump = np.random.uniform(5.0, 12.0, size=len(treated_indices))
            X_sim[treated_indices, 6] = np.clip(X_sim[treated_indices, 6] + comm_bump, 0, 100)
            
            # 25% of compliant students also attain an Internship (Feature index 4)
            internship_chance = np.random.rand(len(treated_indices)) < 0.25
            X_sim[treated_indices[internship_chance], 4] = 1
        
        # Re-predict outcome under intervention
        X_sim_s = scaler.transform(X_sim)
        sim_probs = model.predict_proba(X_sim_s)[:, 1]
        sim_preds = (sim_probs >= 0.50).astype(int)
        
        cohort_yield = np.mean(sim_preds)
        yields.append(cohort_yield)
        
        # Incremental placements over baseline
        addl_placed = np.sum(sim_preds) - np.sum(preds_baseline)
        addl_placed_list.append(addl_placed)
        
        # Financial ROI
        total_intervention_cost = len(treated_indices) * COST_PER_STUDENT
        total_economic_gain = addl_placed * VALUE_PER_PLACEMENT
        net_benefit = total_economic_gain - total_intervention_cost
        cbr = (total_economic_gain / total_intervention_cost) if total_intervention_cost > 0 else 1.0
        
        net_benefits.append(net_benefit)
        cbr_list.append(cbr)
        
    mean_yield = float(np.mean(yields))
    mean_addl = float(np.mean(addl_placed_list))
    mean_net_benefit = float(np.mean(net_benefits))
    mean_cbr = float(np.mean(cbr_list))
    
    simulation_results.append({
        'Compliance_Rate_Alpha': f"{int(alpha * 100)}%",
        'Adherence_Description': (
            'Status Quo (No Intervention)' if alpha == 0.0 else
            'Voluntary Low Opt-In' if alpha == 0.20 else
            'Advisory-Prompted Opt-In' if alpha == 0.40 else
            'Departmental Lab Integration' if alpha == 0.60 else
            'Structured Core Requirement' if alpha == 0.80 else
            'Universal Cohort Adoption'
        ),
        'Cohort_Placement_Yield': f"{mean_yield * 100:.2f}%",
        'Yield_Percentage_Raw': round(mean_yield * 100, 2),
        'Absolute_Yield_Gain': f"+{(mean_yield - base_yield) * 100:.2f}pp",
        'Additional_Students_Placed': int(round(mean_addl)),
        'Treated_At_Risk_Students': int(round(n_at_risk * alpha)),
        'Intervention_Cost_USD': f"${int(round(n_at_risk * alpha * COST_PER_STUDENT)):,}",
        'Economic_Gain_USD': f"${int(round(mean_addl * VALUE_PER_PLACEMENT)):,}",
        'Net_Institutional_ROI_USD': f"${int(round(mean_net_benefit)):,}",
        'Benefit_Cost_Ratio_BCR': f"{mean_cbr:.2f}x"
    })

df_sim = pd.DataFrame(simulation_results)
df_sim.to_csv(os.path.join(OUTPUT_DIR, 'table_exp_j_macro_policy_simulation.csv'), index=False)
print("\n[SUCCESS] Saved macro policy simulation to table_exp_j_macro_policy_simulation.csv")
print(df_sim[['Compliance_Rate_Alpha', 'Cohort_Placement_Yield', 'Absolute_Yield_Gain', 'Additional_Students_Placed', 'Benefit_Cost_Ratio_BCR', 'Net_Institutional_ROI_USD']].to_string(index=False))
