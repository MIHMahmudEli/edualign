"""
Experiment 5: NLP Representation Ablation Benchmark
Comparing Exact Keyword Matching, Word TF-IDF, Character N-Grams, and Latent SVD Projections.
"""

import os
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'csv') if os.path.exists(os.path.join(BASE_DIR, 'datasets', 'csv')) else (os.path.join(BASE_DIR, 'csv') if os.path.exists(os.path.join(BASE_DIR, 'csv')) else BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'derived_matrices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 5: NLP REPRESENTATION ABLATION BENCHMARK")
print("=" * 80)

hf_skills_path = os.path.join(RAW_DIR, 'hf_skill_gap_skills.csv')
academic_path = os.path.join(RAW_DIR, 'hf_job_skill_set.csv')

df_hf_skills = pd.read_csv(hf_skills_path)
df_academic = pd.read_csv(academic_path)

# Extract top 250 market skills
market_skills_list = []
for s in df_hf_skills['job_skills'].dropna():
    for item in str(s).split(','):
        cleaned = item.strip().lower()
        if cleaned:
            market_skills_list.append(cleaned)
top_market_skills = [s for s, _ in Counter(market_skills_list).most_common(250)]

# Extract top 250 academic skills
academic_skills_list = []
for _, row in df_academic.iterrows():
    if pd.notna(row.get('job_skill_set')):
        s_str = str(row['job_skill_set']).strip('[]').replace("'", "").replace('"', '')
        for s in s_str.split(','):
            cleaned = s.strip().lower()
            if cleaned:
                academic_skills_list.append(cleaned)
top_academic_skills = [s for s, _ in Counter(academic_skills_list).most_common(250)]

print(f"Corpus: {len(top_market_skills)} Market Skills vs {len(top_academic_skills)} Academic Skills.")

representations = [
    ('Exact String Match (Lexical Jaccard)', 'exact', None),
    ('Word-Level TF-IDF (1-Gram)', 'word_tfidf', None),
    ('Char N-Gram TF-IDF (1-3 Gram)', 'char_tfidf', None),
    ('Char N-Gram + SVD (d=25)', 'svd', 25),
    ('Char N-Gram + SVD (d=50, Proposed)', 'svd', 50),
    ('Char N-Gram + SVD (d=100)', 'svd', 100),
]

ablation_rows = []

for rep_name, mode, dim in representations:
    t0 = time.time()
    
    if mode == 'exact':
        # Exact string match: 1.0 if identical else 0.0
        set_acad = set(top_academic_skills)
        max_sims = np.array([1.0 if s in set_acad else 0.0 for s in top_market_skills])
        
    elif mode == 'word_tfidf':
        vec = TfidfVectorizer(ngram_range=(1, 1), analyzer='word')
        all_s = top_market_skills + top_academic_skills
        X = vec.fit_transform(all_s)
        X_m = X[:len(top_market_skills)]
        X_a = X[len(top_market_skills):]
        sim_mat = cosine_similarity(X_m, X_a)
        max_sims = np.max(sim_mat, axis=1)
        
    elif mode == 'char_tfidf':
        vec = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb')
        all_s = top_market_skills + top_academic_skills
        X = vec.fit_transform(all_s)
        X_m = X[:len(top_market_skills)]
        X_a = X[len(top_market_skills):]
        sim_mat = cosine_similarity(X_m, X_a)
        max_sims = np.max(sim_mat, axis=1)
        
    elif mode == 'svd':
        vec = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb')
        all_s = top_market_skills + top_academic_skills
        X = vec.fit_transform(all_s)
        svd = TruncatedSVD(n_components=dim, random_state=42)
        X_dense = svd.fit_transform(X)
        norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_dense / norms
        X_m = X_norm[:len(top_market_skills)]
        X_a = X_norm[len(top_market_skills):]
        sim_mat = np.dot(X_m, X_a.T)
        max_sims = np.max(sim_mat, axis=1)
        
    elapsed_ms = (time.time() - t0) * 1000.0
    
    # Metrics
    mean_sim = np.mean(max_sims)
    oov_rate = np.mean(max_sims == 0.0) * 100.0 # Percentage of skills with 0 alignment
    coverage_tau50 = np.mean(max_sims >= 0.50) * 100.0
    std_sim = np.std(max_sims)
    
    ablation_rows.append({
        'Representation_Method': rep_name,
        'Mean_Semantic_Alignment': round(float(mean_sim), 4),
        'OOV_Zero_Match_Rate': f"{oov_rate:.1f}%",
        'Alignment_Coverage_tau05': f"{coverage_tau50:.1f}%",
        'Dispersion_StdDev': round(float(std_sim), 4),
        'Latency_ms': round(float(elapsed_ms), 1),
        'Continuity_Status': 'Discrete / Brittle' if oov_rate > 20 else 'Continuous & Dense'
    })

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(os.path.join(OUTPUT_DIR, 'table_exp5_nlp_representation_ablation.csv'), index=False)

print("Saved: table_exp5_nlp_representation_ablation.csv")
print(df_ablation[['Representation_Method', 'Mean_Semantic_Alignment', 'OOV_Zero_Match_Rate', 'Alignment_Coverage_tau05', 'Continuity_Status']].to_string(index=False))

print("\nEXPERIMENT 5 COMPLETED!")
