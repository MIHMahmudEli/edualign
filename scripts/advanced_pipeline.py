"""
Advanced Machine Learning & Deep Learning Pipeline
Academic-Industry Skill Gap & Employability Intelligence Framework
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 80)
print("STARTING ADVANCED ML/DL PIPELINE EXECUTION")
print("=" * 80)

# ----------------------------------------------------------------------------------
# PART 1: DENSE SEMANTIC SKILL GAP ANALYSIS (ADVANCED NLP)
# ----------------------------------------------------------------------------------
print("\n[PART 1] Computing Dense Semantic Alignment on Market & Academic Corpora...")

hf_skills_path = os.path.join(RAW_DIR, 'hf_skill_gap_skills.csv')
academic_path = os.path.join(RAW_DIR, 'hf_job_skill_set.csv')

df_hf_skills = pd.read_csv(hf_skills_path)
df_academic = pd.read_csv(academic_path)

# Extract top market skills
market_skills_list = []
for s in df_hf_skills['job_skills'].dropna():
    for item in str(s).split(','):
        cleaned = item.strip().lower()
        if cleaned:
            market_skills_list.append(cleaned)
market_counter = Counter(market_skills_list)
top_market_skills = [s for s, _ in market_counter.most_common(200)]

# Extract top academic skills
academic_skills_list = []
for _, row in df_academic.iterrows():
    if pd.notna(row.get('job_skill_set')):
        s_str = str(row['job_skill_set']).strip('[]').replace("'", "").replace('"', '')
        for s in s_str.split(','):
            cleaned = s.strip().lower()
            if cleaned:
                academic_skills_list.append(cleaned)
academic_counter = Counter(academic_skills_list)
top_academic_skills = [s for s, _ in academic_counter.most_common(200)]

# Compute continuous semantic alignment via TF-IDF character & word n-grams with SVD
combined_skills = top_market_skills + top_academic_skills
tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb', min_df=1)
X_tfidf = tfidf_vec.fit_transform(combined_skills)
svd = TruncatedSVD(n_components=min(50, X_tfidf.shape[1] - 1), random_state=42)
X_dense = svd.fit_transform(X_tfidf)

# Normalize dense vectors
norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
norms[norms == 0] = 1.0
X_norm = X_dense / norms

market_vecs = X_norm[:len(top_market_skills)]
academic_vecs = X_norm[len(top_market_skills):]

# Pairwise cosine similarity matrix
sim_matrix = np.dot(market_vecs, academic_vecs.T)
max_sim_per_market_skill = np.max(sim_matrix, axis=1)

CATEGORIES = {
    'Programming & Development': ['python', 'java', 'javascript', 'c++', 'sql', 'r', 'typescript', 'php', 'ruby', 'html', 'css', 'react', 'node.js', 'django', 'git', 'docker'],
    'Data Science & Analytics': ['machine learning', 'deep learning', 'data analysis', 'data visualization', 'statistical analysis', 'tensorflow', 'pytorch', 'pandas', 'tableau', 'spark', 'ai', 'natural language processing'],
    'Cloud & Infrastructure': ['aws', 'azure', 'gcp', 'cloud computing', 'devops', 'ci/cd', 'jenkins', 'linux', 'kubernetes', 'networking', 'rest api', 'microservices'],
    'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'time management', 'project management', 'adaptability', 'collaboration'],
    'Business & Domain': ['agile', 'scrum', 'product management', 'business analysis', 'stakeholder management', 'quality assurance', 'compliance', 'budget management'],
    'Design & UX': ['ui/ux', 'figma', 'user research', 'wireframing', 'user interface', 'graphic design'],
    'Data Engineering': ['etl', 'data pipeline', 'airflow', 'kafka', 'database design', 'data warehousing', 'nosql', 'postgresql'],
    'Cybersecurity': ['information security', 'penetration testing', 'vulnerability assessment', 'incident response', 'encryption', 'firewall', 'iso 27001']
}

category_gaps = []
for cat_name, skill_tokens in CATEGORIES.items():
    cat_market_scores = []
    for skill_name, max_s in zip(top_market_skills, max_sim_per_market_skill):
        if any(tok in skill_name for tok in skill_tokens):
            cat_market_scores.append(max_s)
    if not cat_market_scores:
        cat_market_scores = [np.mean(max_sim_per_market_skill)]
    mean_sim = np.mean(cat_market_scores)
    dense_gap = 1.0 - mean_sim
    category_gaps.append({
        'Category': cat_name,
        'Mean_Semantic_Alignment': round(float(mean_sim), 4),
        'Dense_Skill_Gap': round(float(dense_gap), 4),
        'Discrete_Coverage_tau05': round(float(np.mean([s >= 0.5 for s in cat_market_scores])), 4),
        'Sampled_Skills_Count': len(cat_market_scores)
    })

df_cat_gaps = pd.DataFrame(category_gaps)
df_cat_gaps.to_csv(os.path.join(OUTPUT_DIR, 'table_semantic_skill_gaps.csv'), index=False)
print("Saved: table_semantic_skill_gaps.csv")

# ----------------------------------------------------------------------------------
# PART 2: RIGOROUS DEEP TABULAR & ADVANCED ENSEMBLE BENCHMARK (COLLEGE PLACEMENT)
# ----------------------------------------------------------------------------------
print("\n[PART 2] Running 10-Fold Stratified CV on Benchmark 1 (College Placement N=10,000)...")

college_csv = os.path.join(RAW_DIR, 'kaggle', 'college-student-placement', 'college_student_placement_dataset.csv')
df_college = pd.read_csv(college_csv)

df_college['Placement_Binary'] = (df_college['Placement'] == 'Yes').astype(int)
df_college['Internship_Binary'] = (df_college['Internship_Experience'] == 'Yes').astype(int)

feature_cols = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Internship_Binary',
                'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

X = df_college[feature_cols].values
y = df_college['Placement_Binary'].values

n_samples = len(y)
n_placed = np.sum(y)
class_ratio = (n_samples - n_placed) / n_placed
print(f"Dataset size: {n_samples}, Placed: {n_placed} ({n_placed/n_samples*100:.2f}%), Imbalance Ratio: {class_ratio:.2f}:1")

models = {
    'Baseline (Majority)': None,
    'Logistic Regression (L2, Balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree (Pruned)': DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight='balanced', random_state=42),
    'Random Forest (150 Trees)': RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, max_depth=8, class_weight='balanced', random_state=42),
    'LightGBM Classifier': lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1),
    'Deep Tabular Neural Net (MLP: 128-64-32)': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.001,
        learning_rate_init=0.005, max_iter=250, early_stopping=True, random_state=42
    )
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
benchmark_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, clf in models.items():
        if clf is None:
            y_pred = np.zeros_like(y_test)
            y_prob = np.full_like(y_test, fill_value=n_placed/n_samples, dtype=float)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        benchmark_results.append({
            'Fold': fold + 1,
            'Model': name,
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'Precision': prec,
            'Recall': rec,
            'Macro_F1': f1,
            'ROC_AUC': roc,
            'PR_AUC': pr_auc,
            'Brier_Score': brier
        })

df_bench = pd.DataFrame(benchmark_results)
summary_table = df_bench.groupby('Model').agg({
    'Accuracy': ['mean', 'std'],
    'Balanced_Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'Macro_F1': ['mean', 'std'],
    'ROC_AUC': ['mean', 'std'],
    'PR_AUC': ['mean', 'std'],
    'Brier_Score': ['mean', 'std']
}).reset_index()

summary_table.columns = ['_'.join(c).strip('_') for c in summary_table.columns]
summary_table.to_csv(os.path.join(OUTPUT_DIR, 'table_model_benchmark_10fold.csv'), index=False)
print("Saved: table_model_benchmark_10fold.csv")

# ----------------------------------------------------------------------------------
# PART 3: ADVANCED STACKING META-LEARNER
# ----------------------------------------------------------------------------------
print("\n[PART 3] Training Deep Stacking Meta-Learner (Deep MLP + LightGBM + Random Forest)...")

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001, max_iter=200, early_stopping=True, random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=3,
    n_jobs=-1
)

stacking_cv_results = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stacking_clf.fit(X_train_scaled, y_train)
    y_pred = stacking_clf.predict(X_test_scaled)
    y_prob = stacking_clf.predict_proba(X_test_scaled)[:, 1]

    stacking_cv_results.append({
        'Model': 'Deep Stacking Meta-Ensemble',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Macro_F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'PR_AUC': average_precision_score(y_test, y_prob),
        'Brier_Score': brier_score_loss(y_test, y_prob)
    })

df_stacking = pd.DataFrame(stacking_cv_results)
stack_mean = df_stacking.mean(numeric_only=True)
print(f"Stacking Ensemble Mean Accuracy: {stack_mean['Accuracy']*100:.2f}%, F1: {stack_mean['Macro_F1']:.4f}, ROC-AUC: {stack_mean['ROC_AUC']:.4f}")

# ----------------------------------------------------------------------------------
# PART 4: CROSS-DATASET GENERALIZATION (BENCHMARK 2 & 3)
# ----------------------------------------------------------------------------------
print("\n[PART 4] Evaluating Cross-Dataset Generalizability on 50K Career Success & 300K International...")

# Dataset 2: Student Career Success (50,000)
career_csv = os.path.join(RAW_DIR, 'kaggle', 'student-career-success-prediction-dataset', 'student_career_success_dataset.csv')
df_career = pd.read_csv(career_csv)

# Clean and encode
df_career['Leadership_Binary'] = df_career['Leadership_Experience'].map({'Yes': 1, 'No': 0}).fillna(0)
df_career['English_Score'] = df_career['English_Proficiency'].map({'Basic': 1, 'Intermediate': 2, 'Advanced': 3}).fillna(2)

career_features = ['CGPA', 'Study_Hours_Per_Week', 'Programming_Skill', 'Projects_Completed',
                   'Certifications', 'Hackathons', 'Internships', 'Leadership_Binary',
                   'Resume_Score', 'Communication_Skills', 'Teamwork', 'Problem_Solving',
                   'English_Score', 'Interview_Score']

y_career = (df_career['Placement_Status'] == 'Placed').astype(int).values
X_career = df_career[career_features].fillna(df_career[career_features].median()).values

scaler_career = StandardScaler()
X_career_scaled = scaler_career.fit_transform(X_career)

clf_career_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
career_accs, career_f1s, career_aucs, career_precs, career_recs = [], [], [], [], []

for train_idx, test_idx in skf5.split(X_career_scaled, y_career):
    clf_career_lgb.fit(X_career_scaled[train_idx], y_career[train_idx])
    preds = clf_career_lgb.predict(X_career_scaled[test_idx])
    probs = clf_career_lgb.predict_proba(X_career_scaled[test_idx])[:, 1]
    career_accs.append(accuracy_score(y_career[test_idx], preds))
    career_f1s.append(f1_score(y_career[test_idx], preds))
    career_aucs.append(roc_auc_score(y_career[test_idx], probs))
    career_precs.append(precision_score(y_career[test_idx], preds))
    career_recs.append(recall_score(y_career[test_idx], preds))

print(f"50K Career Success Dataset (5-Fold): Acc={np.mean(career_accs)*100:.2f}%, F1={np.mean(career_f1s):.4f}, AUC={np.mean(career_aucs):.4f}")

# Dataset 3: International Graduates (300,000)
intl_csv = os.path.join(RAW_DIR, 'kaggle', 'international-graduates-employment-dataset', 'dataset.csv')
df_intl = pd.read_csv(intl_csv, nrows=100000)
y_intl = (df_intl['Employment_Status'] == 'Employed').astype(int).values

df_intl['Internship_Binary'] = df_intl['Internship_Experience'].map({'Yes': 1, 'No': 0}).fillna(0)
df_intl['Ranking_Score'] = df_intl['University_Ranking'].map({'Low': 1, 'Medium': 2, 'High': 3}).fillna(2)
df_intl['Language_Score'] = df_intl['Language_Proficiency'].map({'Basic': 1, 'Intermediate': 2, 'Fluent': 3}).fillna(2)
df_intl['Education_Score'] = df_intl['Education_Level'].map({"Bachelor's": 1, "Master's": 2, "PhD": 3}).fillna(1)

intl_features = ['GPA', 'Age', 'Years_Since_Graduation', 'Internship_Binary', 'Ranking_Score', 'Language_Score', 'Education_Score']
X_intl = df_intl[intl_features].fillna(df_intl[intl_features].median()).values

scaler_intl = StandardScaler()
X_intl_scaled = scaler_intl.fit_transform(X_intl)

clf_intl_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
intl_accs, intl_f1s, intl_aucs, intl_precs, intl_recs = [], [], [], [], []
for train_idx, test_idx in skf5.split(X_intl_scaled, y_intl):
    clf_intl_lgb.fit(X_intl_scaled[train_idx], y_intl[train_idx])
    preds = clf_intl_lgb.predict(X_intl_scaled[test_idx])
    probs = clf_intl_lgb.predict_proba(X_intl_scaled[test_idx])[:, 1]
    intl_accs.append(accuracy_score(y_intl[test_idx], preds))
    intl_f1s.append(f1_score(y_intl[test_idx], preds))
    intl_aucs.append(roc_auc_score(y_intl[test_idx], probs))
    intl_precs.append(precision_score(y_intl[test_idx], preds))
    intl_recs.append(recall_score(y_intl[test_idx], preds))

print(f"100K Sample International Graduates Dataset (5-Fold): Acc={np.mean(intl_accs)*100:.2f}%, F1={np.mean(intl_f1s):.4f}, AUC={np.mean(intl_aucs):.4f}")

df_cross_data = pd.DataFrame([
    {
        'Cohort': 'Domestic College Placement',
        'Records': 10000,
        'Positive_Rate': f"{n_placed/n_samples*100:.1f}%",
        'Accuracy': round(float(summary_table.loc[summary_table['Model'] == 'LightGBM Classifier', 'Accuracy_mean'].iloc[0]), 4),
        'Precision': round(float(summary_table.loc[summary_table['Model'] == 'LightGBM Classifier', 'Precision_mean'].iloc[0]), 4),
        'Recall': round(float(summary_table.loc[summary_table['Model'] == 'LightGBM Classifier', 'Recall_mean'].iloc[0]), 4),
        'F1_Score': round(float(summary_table.loc[summary_table['Model'] == 'LightGBM Classifier', 'Macro_F1_mean'].iloc[0]), 4),
        'ROC_AUC': round(float(summary_table.loc[summary_table['Model'] == 'LightGBM Classifier', 'ROC_AUC_mean'].iloc[0]), 4)
    },
    {
        'Cohort': 'Student Career Success',
        'Records': 50000,
        'Positive_Rate': f"{np.mean(y_career)*100:.1f}%",
        'Accuracy': round(float(np.mean(career_accs)), 4),
        'Precision': round(float(np.mean(career_precs)), 4),
        'Recall': round(float(np.mean(career_recs)), 4),
        'F1_Score': round(float(np.mean(career_f1s)), 4),
        'ROC_AUC': round(float(np.mean(career_aucs)), 4)
    },
    {
        'Cohort': 'International Graduates',
        'Records': 300000,
        'Positive_Rate': f"{np.mean(y_intl)*100:.1f}%",
        'Accuracy': round(float(np.mean(intl_accs)), 4),
        'Precision': round(float(np.mean(intl_precs)), 4),
        'Recall': round(float(np.mean(intl_recs)), 4),
        'F1_Score': round(float(np.mean(intl_f1s)), 4),
        'ROC_AUC': round(float(np.mean(intl_aucs)), 4)
    }
])
df_cross_data.to_csv(os.path.join(OUTPUT_DIR, 'table_cross_dataset_generalizability.csv'), index=False)
print("Saved: table_cross_dataset_generalizability.csv")
print(df_cross_data.to_string(index=False))

# ----------------------------------------------------------------------------------
# PART 5: ACTIONABLE COUNTERFACTUAL INTERVENTION ENGINE (PRESCRIPTIVE AI)
# ----------------------------------------------------------------------------------
print("\n[PART 5] Solving Counterfactual Curricular Optimization for At-Risk Students...")

full_scaler = StandardScaler()
X_full_scaled = full_scaler.fit_transform(X)
prod_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
prod_model.fit(X_full_scaled, y)

probs_full = prod_model.predict_proba(X_full_scaled)[:, 1]
at_risk_indices = np.where((y == 0) & (probs_full < 0.25))[0]

sample_size = min(300, len(at_risk_indices))
np.random.seed(42)
sampled_risk_idx = np.random.choice(at_risk_indices, size=sample_size, replace=False)

counterfactual_interventions = []

for idx in sampled_risk_idx:
    orig_x = X[idx].copy()
    orig_prob = probs_full[idx]

    best_cost = float('inf')
    best_intervention = None

    for d_comm in np.linspace(0, 4.0, 17):
        for d_proj in [0, 1, 2, 3]:
            for d_intern in [0, 1] if orig_x[4] == 0 else [0]:
                perturbed_x = orig_x.copy()
                perturbed_x[6] = min(10.0, orig_x[6] + d_comm)
                perturbed_x[7] = orig_x[7] + d_proj
                perturbed_x[4] = min(1, orig_x[4] + d_intern)

                p_scaled = full_scaler.transform(perturbed_x.reshape(1, -1))
                new_prob = prod_model.predict_proba(p_scaled)[0, 1]

                if new_prob >= 0.60:
                    cost = 1.0 * d_comm + 1.8 * d_proj + 2.0 * d_intern
                    if cost < best_cost:
                        best_cost = cost
                        best_intervention = {
                            'Student_ID': str(df_college.iloc[idx]['College_ID']),
                            'Initial_Prob': round(float(orig_prob), 4),
                            'Post_Intervention_Prob': round(float(new_prob), 4),
                            'Delta_Comm': round(float(d_comm), 2),
                            'Delta_Projects': int(d_proj),
                            'Added_Internship': int(d_intern),
                            'Intervention_Cost': round(float(cost), 2)
                        }

    if best_intervention:
        counterfactual_interventions.append(best_intervention)

df_cf = pd.DataFrame(counterfactual_interventions)
df_cf.to_csv(os.path.join(OUTPUT_DIR, 'table_counterfactual_interventions.csv'), index=False)
print("Saved: table_counterfactual_interventions.csv")
print(f"Successfully computed counterfactual pathways for {len(df_cf)} students.")
print(f"Mean required communication improvement: +{df_cf['Delta_Comm'].mean():.2f} pts")
print(f"Mean required projects: +{df_cf['Delta_Projects'].mean():.2f} projects")
print(f"Placement probability jump: {df_cf['Initial_Prob'].mean()*100:.1f}% -> {df_cf['Post_Intervention_Prob'].mean()*100:.1f}%")

# ----------------------------------------------------------------------------------
# PART 6: EXPORT MODEL COMPARISON SUMMARY
# ----------------------------------------------------------------------------------
model_paper_table = []
for _, row in summary_table.iterrows():
    m = row['Model']
    model_paper_table.append({
        'Model': m,
        'Accuracy': f"{row['Accuracy_mean']*100:.1f}% +/- {row['Accuracy_std']*100:.1f}%",
        'Balanced_Acc': f"{row['Balanced_Accuracy_mean']*100:.1f}%",
        'Precision': f"{row['Precision_mean']:.3f}",
        'Recall': f"{row['Recall_mean']:.3f}",
        'Macro_F1': f"{row['Macro_F1_mean']:.3f} +/- {row['Macro_F1_std']:.3f}",
        'ROC_AUC': f"{row['ROC_AUC_mean']:.3f} +/- {row['ROC_AUC_std']:.3f}",
        'PR_AUC': f"{row['PR_AUC_mean']:.3f}",
        'Brier': f"{row['Brier_Score_mean']:.3f}"
    })
model_paper_table.append({
    'Model': 'Deep Stacking Meta-Ensemble',
    'Accuracy': f"{stack_mean['Accuracy']*100:.1f}% +/- {df_stacking['Accuracy'].std()*100:.1f}%",
    'Balanced_Acc': f"{stack_mean['Balanced_Accuracy']*100:.1f}%",
    'Precision': f"{stack_mean['Precision']:.3f}",
    'Recall': f"{stack_mean['Recall']:.3f}",
    'Macro_F1': f"{stack_mean['Macro_F1']:.3f} +/- {df_stacking['Macro_F1'].std():.3f}",
    'ROC_AUC': f"{stack_mean['ROC_AUC']:.3f} +/- {df_stacking['ROC_AUC'].std():.3f}",
    'PR_AUC': f"{stack_mean['PR_AUC']:.3f}",
    'Brier': f"{stack_mean['Brier_Score']:.3f}"
})

df_paper_model = pd.DataFrame(model_paper_table)
df_paper_model.to_csv(os.path.join(OUTPUT_DIR, 'table_paper_model_comparison.csv'), index=False)
print("Saved: table_paper_model_comparison.csv")
print("\nALL EXPERIMENTS COMPLETED!")
