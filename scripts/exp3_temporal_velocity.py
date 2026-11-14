"""
Experiment 3: Longitudinal Skill Momentum Velocity & Curricular Decay Modeling (2020-2026)
Quantifying the rate of skill obsolescence and curricular lag over a 4-year degree cycle.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 3: TEMPORAL SKILL VELOCITY & CURRICULUM DECAY (2020-2026)")
print("=" * 80)

ai_csv = os.path.join(RAW_DIR, 'kaggle', 'ai-and-data-science-job-market-dataset-20202026', 'AI Job Market Dataset.csv')
df = pd.read_csv(ai_csv)

skill_cols = ['skills_python', 'skills_sql', 'skills_ml', 'skills_deep_learning', 'skills_cloud']
skill_names = {
    'skills_python': 'Python Programming',
    'skills_sql': 'SQL & Relational Databases',
    'skills_ml': 'Machine Learning Core',
    'skills_deep_learning': 'Deep Learning & Neural Networks',
    'skills_cloud': 'Cloud Infrastructure & DevOps'
}

years = sorted(df['job_posting_year'].unique())
yearly_stats = []

for y in years:
    sub = df[df['job_posting_year'] == y]
    n_jobs = len(sub)
    row = {'Year': y, 'Total_Postings': n_jobs}
    for col in skill_cols:
        rate = sub[col].sum() / n_jobs
        row[col] = rate
    yearly_stats.append(row)

df_yearly = pd.DataFrame(yearly_stats)

# Compute Trend Slopes, CAGRs, and Curricular Latency
metrics_summary = []

for col in skill_cols:
    rates = df_yearly[col].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, rates)
    
    # 2020 rate vs 2026 rate
    r_2020 = df_yearly.loc[df_yearly['Year'] == 2020, col].iloc[0]
    r_2026 = df_yearly.loc[df_yearly['Year'] == 2026, col].iloc[0]
    
    # CAGR
    cagr = ((r_2026 / r_2020) ** (1.0 / 6.0) - 1.0) * 100
    
    # 4-Year Curricular Decay (Divergence between year 0 baseline and year 4 reality)
    r_2024 = df_yearly.loc[df_yearly['Year'] == 2024, col].iloc[0]
    decay_4yr = abs(r_2024 - r_2020) / r_2020 * 100
    
    metrics_summary.append({
        'Skill_Competency': skill_names[col],
        'Penetration_2020': f"{r_2020*100:.1f}%",
        'Penetration_2026': f"{r_2026*100:.1f}%",
        'Annual_Velocity_Slope': round(float(slope), 4),
        'CAGR_Percent': round(float(cagr), 2),
        'P_Value': round(float(p_value), 4),
        'Curricular_Lag_4Yr_Divergence': f"{decay_4yr:.1f}%",
        'Trajectory': 'Expanding' if slope > 0 else 'Contracting' if slope < 0 else 'Stationary'
    })

df_metrics = pd.DataFrame(metrics_summary)
df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'table_exp3_temporal_skill_decay.csv'), index=False)
df_yearly.to_csv(os.path.join(OUTPUT_DIR, 'table_exp3_yearly_penetration_rates.csv'), index=False)

print("Saved: table_exp3_temporal_skill_decay.csv")
print("Saved: table_exp3_yearly_penetration_rates.csv")
print(df_metrics.to_string(index=False))

print("\nEXPERIMENT 3 COMPLETED!")
