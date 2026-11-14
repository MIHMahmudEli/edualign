# EduAlign: A Deep Stacking and Counterfactual Framework for Curricular Gap Analytics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Replication-18 Modules](https://img.shields.io/badge/Replication-18%20Modules%20Deterministic-brightgreen.svg)](scripts/run_all_experiments.py)
[![Matrices-43 Benchmark CSVs](https://img.shields.io/badge/Matrices-43%20Benchmark%20CSVs-orange.svg)](derived_matrices/)
[![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![EEOC Compliant](https://img.shields.io/badge/Fairness-EEOC%20Four--Fifths%20Compliant-purple.svg)](scripts/exp1_fairness_audit.py)

This repository constitutes the official open-source replication package, empirical benchmark matrices, and experimental pipelines for the research study:

> **"Bridging the Academic–Industry Skills Divide: A Multi-Source Deep Learning Framework for Dense Semantic Gap Quantification and Prescriptive Employability Intelligence"**

---

## 1. Executive Summary & Research Scope

The structural divergence between higher education curricula and labor market competency demands represents a critical macroeconomic challenge. Traditional survey-based institutional assessments suffer from severe recall bias and multi-year reporting latency. 

This research establishes a scalable, data-driven framework that harmonizes **15 heterogeneous datasets comprising 3,176,266 records** across real-time industry vacancies and multi-cohort graduate registries.

```
       [ Stage 1: Ingestion & Longitudinal Velocity ]
        15 Datasets (N = 3,176,266) | dv/dt Skill Momentum Gradients
                              │
                              ▼
       [ Stage 2: Dense Latent Semantic Representation ]
        Sub-Word Char n-Gram SVD (d = 50) | 0.0% OOV Disconnect
                              │
                              ▼
       [ Stage 3: Tabular Meta-Learning Benchmark ]
        Deep Stacking Meta-Ensemble (100% Recall, 99.82% Precision, AUC 1.000)
                              │
                              ▼
       [ Stage 4: Second-Order Explainability & Theory ]
        TreeSHAP Feature Attribution | Discovery of Compensatory Capital
                              │
                              ▼
       [ Stage 5: Prescriptive Counterfactual Recourse ]
        Constrained L0 Pareto Sparsity | Feasible Semestral Time Budgets (<=135h)
                              │
                              ▼
       [ Stage 6: Ethical Governance & Macro Policy Simulation ]
        US EEOC Intersectional Parity (DIR >= 0.9922) | 9.8x Benefit-Cost Ratio
```

### Core Methodological Contributions:
1. **Multi-Source Ingestion & Longitudinal Velocity:** Harmonizes 15 multi-national corpora to quantify annual skill momentum velocity gradients ($dv/dt$) and exponential half-life decay models across a 4-year degree cycle.
2. **Continuous Dense Semantic Representation:** Maps syllabus curricula and market requirements into a continuous sub-word character $n$-gram SVD latent space ($d=50$), achieving **0.0% out-of-vocabulary (OOV)** disconnect with a 2.8 MB memory footprint.
3. **Deep Tabular Stacking Meta-Ensemble:** Solves severe class imbalance ($5.03:1$ to $20:1$ scarcity) across 360,000 multi-cohort students, achieving **100% minority recall, 99.82% precision, and 1.000 ROC-AUC** with non-parametric statistical guarantees (Wilcoxon $p < 0.001$, Cohen's $d = 30.54$).
4. **Compensatory Capital Theory:** Identifies through second-order TreeSHAP interactions that verified practical capstone deliverables and interpersonal oral communication deliver strong positive log-odds uplifts that systematically offset classroom GPA deficits.
5. **Prescriptive Counterfactual Recourse:** Formulates a constrained optimization algorithm under $L_0$ Pareto sparsity, proving that at-risk students can achieve target placement with actionable pedagogical adjustments within a realistic 135-hour semestral budget while ensuring 98.72% data manifold inlier compliance.
6. **Algorithmic Governance & Macro Policy ROI:** Demonstrates compliance with the US EEOC four-fifths rule across four intersectional demographic strata ($\text{DIR} \ge 0.9922$, $\text{EOD} \le 0.0021$) and simulates university-wide adoption across 10,000 undergraduates (+3,370 placed graduates, $9.8\times$ Benefit-Cost Ratio, +\$25.7M Net Institutional ROI).

---

## 2. Repository Layout

```text
edualign/
├── derived_matrices/                  # 43 Canonical reproducible benchmark output matrices + summary
│   ├── exp_a_statistical_summary.txt # Non-parametric statistical summary (Wilcoxon & bootstrap)
│   ├── table1_descriptive_statistics.csv # Baseline descriptive stats (t-stat, p-val, Cohen's d)
│   ├── table2_chi_square_tests.csv    # Chi-square tests of independence & Cramer's V
│   ├── table3_correlation_matrix.csv  # Pearson correlation matrix across feature dimensions
│   ├── table4_logistic_regression.csv # Logistic regression coefficients and odds ratios
│   ├── table5_random_forest_importance.csv # Random Forest Gini impurity feature importances
│   ├── table_confusion_matrices.csv   # Confusion matrix decomposition across all 7 models
│   ├── table_counterfactual_interventions.csv # Individual student recourse pathways (N=83)
│   ├── table_cross_dataset_generalizability.csv # 3-cohort generalizability benchmark (N=360K)
│   ├── table_deep_mlp_training_history.csv # 50-epoch training & validation loss/acc traces
│   ├── table_exp1_algorithmic_fairness_audit.csv # EEOC four-fifths demographic parity audit
│   ├── table_exp2_cgpa_projects_pdp_grid.csv # Partial dependence 2D grid (CGPA x Projects)
│   ├── table_exp2_compensatory_exchange_rate.csv # Academic-practical compensatory exchange rates
│   ├── table_exp2_compensatory_regimes.csv # Compensatory capital pedagogical regimes (I, II, III)
│   ├── table_exp2_compensatory_surface.csv # 2D communication-project placement surface
│   ├── table_exp3_temporal_skill_decay.csv # Longitudinal velocity gradients and 4-year decay
│   ├── table_exp3_yearly_penetration_rates.csv # Yearly competency penetration rates (2020-2026)
│   ├── table_exp4_noise_stress_testing.csv # Classifier stability under Gaussian noise (0-30%)
│   ├── table_exp5_nlp_representation_ablation.csv # Sub-word SVD vs Word2Vec vs TF-IDF ablation
│   ├── table_exp_a_bootstrap_ci.csv   # 10,000-fold bootstrap 95% confidence intervals
│   ├── table_exp_a_statistical_tests.csv # Pairwise Wilcoxon signed-rank tests vs baselines
│   ├── table_exp_b_covariate_shift_metrics.csv # Wasserstein distance covariate shift metrics
│   ├── table_exp_b_few_shot_adaptation_curve.csv # Few-shot target domain calibration trajectory
│   ├── table_exp_b_zero_shot_transfer_matrix.csv # Cross-cohort zero-shot transfer matrix (3x3)
│   ├── table_exp_c_faithfulness_auc.csv # Feature attribution faithfulness degradation AUC
│   ├── table_exp_c_faithfulness_curves.csv # Prediction degradation under masking (k=1..8)
│   ├── table_exp_c_shap_feature_importance.csv # Mean absolute TreeSHAP feature attribution
│   ├── table_exp_c_shap_interaction_matrix_abs.csv # 2nd-order TreeSHAP pairwise interaction matrix
│   ├── table_exp_c_shap_interaction_matrix_signed.csv # Signed interaction matrix (synergies)
│   ├── table_exp_c_shap_pairwise_ranking.csv # Top synergistic feature interaction ranking
│   ├── table_exp_d_counterfactual_sparsity.csv # L0 sparsity distribution (k active levers)
│   ├── table_exp_d_pareto_actionability_frontier.csv # Pareto frontier across confidence tau
│   ├── table_exp_e_computational_scalability.csv # SLA inference latency, peak RAM, disk size
│   ├── table_exp_f_counterfactual_plausibility.csv # Manifold plausibility sample audit (N=235)
│   ├── table_exp_f_manifold_audit.csv # LOF inlier rate and Mahalanobis distance audit
│   ├── table_exp_f_time_budget.csv    # Cumulative student semestral effort budget (hours)
│   ├── table_exp_g_transformer_benchmark.csv # Transformer vs SVD representation efficiency
│   ├── table_exp_h_intersectional_fairness.csv # Multi-group fairness (Gender x Origin tiers)
│   ├── table_exp_i_label_noise_imbalance.csv # 72-cell factorial noise and imbalance benchmark
│   ├── table_exp_j_macro_policy_simulation.csv # Policy placement yield Pi(alpha) & Net ROI ($)
│   ├── table_model_benchmark_10fold.csv # Stratified 10-fold CV raw performance statistics
│   ├── table_model_roc_curves.csv     # 153 operating threshold points across all architectures
│   ├── table_paper_model_comparison.csv # Publication-formatted model benchmark with std devs
│   └── table_semantic_skill_gaps.csv  # Continuous latent semantic gap across 8 categories
│
├── scripts/                           # 24 Self-contained replication pipelines
│   ├── run_all_experiments.py         # Master CLI replication harness (Modules 1–18)
│   ├── advanced_pipeline.py           # Core Deep Stacking Meta-Ensemble benchmark
│   ├── data_loader.py                 # Multi-source data loader and schema normalizer
│   ├── skill_extractor.py             # Sub-word character n-gram & latent SVD engine
│   ├── analysis.py                    # Exploratory analysis and statistical aggregation
│   ├── statistical_tests.py           # Module 01: Classical baseline tests & bivariate stats
│   ├── generate_roc_and_confusion_matrices.py # Module 03: ROC curves & confusion matrices
│   ├── exp_a_statistical_significance.py # Module 04: Wilcoxon tests & bootstrap 95% CIs
│   ├── exp_b_cross_cohort_transfer.py # Module 05: Zero-shot cross-cohort domain transfer
│   ├── exp_c_shap_interaction_faithfulness.py # Module 06: TreeSHAP interaction synergies
│   ├── exp2_compensatory_interaction.py# Module 07: Compensatory Capital surface
│   ├── exp_d_counterfactual_sparsity_pareto.py# Module 08: Pareto recourse optimization
│   ├── exp_e_computational_scalability.py# Module 09: Enterprise serving SLA latency
│   ├── exp1_fairness_audit.py         # Module 10: Algorithmic fairness & EEOC audit
│   ├── exp3_temporal_velocity.py      # Module 11: Longitudinal skill momentum velocity
│   ├── exp4_noise_sensitivity.py      # Module 12: Gaussian covariate perturbation
│   ├── exp5_nlp_ablation.py           # Module 13: NLP representation ablation
│   ├── run_module4_noise_imbalance.py # Module 14: Asymmetric label noise stress-testing
│   ├── run_module5_policy_simulation.py# Module 15: Macro-cohort policy simulation & ROI
│   ├── exp_f_manifold_time_budget.py  # Module 16: Manifold plausibility & time-budget audit
│   ├── exp_g_transformer_benchmark.py # Module 17: Cross-encoder transformer benchmark
│   ├── exp_h_intersectional_fairness.py# Module 18: Intersectional demographic fairness audit
│   ├── generate_all_paper_figures.py  # Master deterministic paper figure generator
│   └── plot_paper_figures_09_10_14.py # High-resolution plotting for trajectory & attribution figures
│
├── datasets/                          # Open-access reference taxonomies
│   ├── canonical_skill_taxonomy.csv   # Standardized 8-category skill taxonomy dictionary
│   └── canonical_skill_taxonomy.json  # Machine-readable JSON taxonomy schema
│
├── .gitignore                         # Clean exclusion rules (Python, OS, temp, datasets)
├── LICENSE                            # Open-source MIT License
├── README.md                          # Repository replication guide & documentation (this file)
├── references.bib                     # Full bibliographic metadata
└── requirements.txt                   # Minimal Python runtime dependencies
```

---

## 3. Systematic Dataset Manifest ($N = 3,176,266$)

All 15 empirical datasets utilized in the paper are summarized below:

| # | Dataset Identity | Citation Key | Source | Records ($N$) | Analytical Domain |
|:---:|---|---|---|---:|---|
| 1 | LinkedIn Jobs & Skills | `asaniczka2024linkedin` | Kaggle | 1,348,510 | Industry Demand & Competency Distribution |
| 2 | HF Skill Gap Jobs | `hf_skillgap_jobs2024` | HuggingFace | 519,841 | Real-Time Job Requisitions |
| 3 | HF Skill Gap Skills | `hf_skillgap_skills2024` | HuggingFace | 519,814 | Machine-Readable Competency Vectors |
| 4 | USA Job Market Skills | `promptcloud2024usajobs` | Kaggle | 316,279 | Regional Tech Roles & Title Matching |
| 5 | International Graduates | `quackquackrp2024international` | Kaggle | 300,000 | Global Career Transitions & Visa Tracking |
| 6 | Student Career Success | `fatima2024student` | Kaggle | 50,000 | Longitudinal Multi-Factor Academic Outcomes |
| 7 | Career Change Prediction | `paliwal2024careerchange` | Kaggle | 38,444 | Cross-Field Career Transition Modeling |
| 8 | Global AI Market Trends | `sajjad2025globalai` | Kaggle | 15,000 | AI/ML Salary & Technical Demand Metrics |
| 9 | HF Job Skill Set | `hf_jobskillset2024` | HuggingFace | 13,833 | Structured Academic Syllabus Tagging |
| 10 | India Skills Intelligence | `indiasp2024skills` | Kaggle | 12,000 | Emerging Economy Technical Profiles |
| 11 | AI Trends (2020–2026) | `aitrends2024market` | Kaggle | 10,345 | Longitudinal Technology Evolution |
| 12 | Domestic College Placement | `sumedh2024college` | Kaggle | 10,000 | Academic Placement Benchmark Records |
| 13 | Global Student Placement | `rakesh2025globalplacement` | Kaggle | 10,000 | Multi-Nation Educational Tracking |
| 14 | NGESS Graduate Survey | `dayo2024ngess` | Kaggle | 7,200 | Developing Economy Workforce Outcomes |
| 15 | India Fresher Hiring | `fresher2024india` | Kaggle | 5,000 | Entry-Level Engineering Placement Records |
| **Total** | **Multi-Source Benchmark Suite** | — | — | **3,176,266** | **Global Academic–Industry Labor Convergence** |

> **Note on Raw Data Access & GitHub Size Limits:**  
> The raw multi-source corpora total **over 6.5 GB** (with individual raw LinkedIn job vacancy dumps exceeding 4.8 GB), which exceeds GitHub's 100 MB per-file upload policy. To ensure instant, frictionless replication without requiring multi-gigabyte downloads, **all canonical pre-processed feature matrices** are hosted directly in [`derived_matrices/`](derived_matrices/). Reviewers who wish to reconstruct the raw ETL from scratch can download the original open-access corpora from their respective Kaggle and HuggingFace public repositories listed above.

---

## 4. Manuscript Table & Figure to Replication Mapping

Every table and figure in the journal manuscript maps directly to a deterministic Python script and canonical CSV matrix in this repository:

### Tables (All 20 Tables in Manuscript)

| Manuscript Table | Description | Generating Script | Source Matrix in `derived_matrices/` |
|:---:|---|---|---|
| **Table 1** | Systematic Summary of 15 Empirical Datasets ($N=3,176,266$) | `scripts/data_loader.py` | Provenance manifest |
| **Table 2** | Continuous Latent Semantic Skill Gap Across Categories | `scripts/advanced_pipeline.py` | `table_semantic_skill_gaps.csv` |
| **Table 3** | 10-Fold CV Performance Benchmark (Domestic College) | `scripts/advanced_pipeline.py` | `table_model_benchmark_10fold.csv`, `table_paper_model_comparison.csv` |
| **Table 4** | Confusion Matrix Decomposition Across All 7 Models | `scripts/generate_roc_and_confusion_matrices.py` | `table_confusion_matrices.csv` |
| **Table 5** | Cross-Cohort Generalization Benchmark Across Registries | `scripts/advanced_pipeline.py` | `table_cross_dataset_generalizability.csv` |
| **Table 6** | Algorithmic Fairness & Demographic Parity Audit | `scripts/exp1_fairness_audit.py` | `table_exp1_algorithmic_fairness_audit.csv` |
| **Table 7** | Compensatory Capital Regimes & Pedagogical Pathways | `scripts/exp2_compensatory_interaction.py` | `table_exp2_compensatory_regimes.csv`, `table_exp2_compensatory_surface.csv` |
| **Table 8** | Longitudinal Skill Momentum Velocity Gradients ($dv/dt$) | `scripts/exp3_temporal_velocity.py` | `table_exp3_temporal_skill_decay.csv`, `table_exp3_yearly_penetration_rates.csv` |
| **Table 9** | Gaussian Covariate Noise Sensitivity & Classifier Resilience | `scripts/exp4_noise_sensitivity.py` | `table_exp4_noise_stress_testing.csv` |
| **Table 10** | NLP Representation Ablation: Lexical vs. Dense Latent SVD | `scripts/exp5_nlp_ablation.py` | `table_exp5_nlp_representation_ablation.csv` |
| **Table 11** | Non-Parametric Significance (Wilcoxon) & Bootstrap 95% CIs | `scripts/exp_a_statistical_significance.py` | `table_exp_a_statistical_tests.csv`, `table_exp_a_bootstrap_ci.csv` |
| **Table 12** | Cross-Cohort Zero-Shot Transfer & Covariate Shift Metrics | `scripts/exp_b_cross_cohort_transfer.py` | `table_exp_b_zero_shot_transfer_matrix.csv`, `table_exp_b_covariate_shift_metrics.csv` |
| **Table 13** | Second-Order TreeSHAP Feature Interaction Synergies | `scripts/exp_c_shap_interaction_faithfulness.py` | `table_exp_c_shap_interaction_matrix_abs.csv`, `table_exp_c_shap_pairwise_ranking.csv` |
| **Table 14** | Counterfactual $L_0$ Sparsity & Pareto Recourse Frontier | `scripts/exp_d_counterfactual_sparsity_pareto.py` | `table_exp_d_counterfactual_sparsity.csv`, `table_exp_d_pareto_actionability_frontier.csv` |
| **Table 15** | Computational Complexity, Peak Memory, & Serving Latency | `scripts/exp_e_computational_scalability.py` | `table_exp_e_computational_scalability.csv` |
| **Table 16** | Counterfactual Manifold Plausibility & Time-Budget Feasibility | `scripts/exp_f_manifold_time_budget.py` | `table_exp_f_manifold_audit.csv`, `table_exp_f_time_budget.csv` |
| **Table 17** | Cross-Encoder & Transformer Representation Benchmark | `scripts/exp_g_transformer_benchmark.py` | `table_exp_g_transformer_benchmark.csv` |
| **Table 18** | Intersectional Multi-Group Fairness & Equalized Odds Audit | `scripts/exp_h_intersectional_fairness.py` | `table_exp_h_intersectional_fairness.csv` |
| **Table 19** | Model Robustness Under Asymmetric Label Noise & Imbalance | `scripts/run_module4_noise_imbalance.py` | `table_exp_i_label_noise_imbalance.csv` |
| **Table 20** | Macroeconomic Institutional Policy Simulation ($9.8\times$ BCR) | `scripts/run_module5_policy_simulation.py` | `table_exp_j_macro_policy_simulation.csv` |

### Figures (All 21 Figures in Manuscript)

| Manuscript Figure | Description | Generating Script | Underlying Data Source |
|:---:|---|---|---|
| **Fig. 1** | Conceptual End-to-End System Architecture | Vector Schematic | System architecture design |
| **Fig. 2** | Deep Stacking Meta-Ensemble Block Diagram | Vector Schematic | Stacking framework schematic |
| **Fig. 3** | Top 10 In-Demand Industry Competencies | `scripts/generate_all_paper_figures.py` | `table_semantic_skill_gaps.csv` |
| **Fig. 4** | Continuous Latent Semantic Skill Gap Plot | `scripts/generate_all_paper_figures.py` | `table_semantic_skill_gaps.csv` |
| **Fig. 5** | Pearson Correlation Heatmap Matrix | `scripts/generate_all_paper_figures.py` | `table3_correlation_matrix.csv` |
| **Fig. 6** | Model Benchmark Cross-Validation Bars | `scripts/generate_all_paper_figures.py` | `table_model_benchmark_10fold.csv` |
| **Fig. 7** | Multi-Model ROC Curves & Zoomed Inset | `scripts/generate_roc_and_confusion_matrices.py` | `table_model_roc_curves.csv` |
| **Fig. 8** | Confusion Matrices Across Evaluated Architectures | `scripts/generate_roc_and_confusion_matrices.py` | `table_confusion_matrices.csv` |
| **Fig. 9** | Deep Tabular MLP 50-Epoch Convergence Trajectories | `scripts/plot_paper_figures_09_10_14.py` | `table_deep_mlp_training_history.csv` |
| **Fig. 10** | Cross-Cohort Transferability Comparison Bars | `scripts/plot_paper_figures_09_10_14.py` | `table_cross_dataset_generalizability.csv` |
| **Fig. 11** | Prescriptive Counterfactual Recourse Scatter | `scripts/generate_all_paper_figures.py` | `table_counterfactual_interventions.csv` |
| **Fig. 12** | Few-Shot Calibration Domain Adaptation Trajectories | `scripts/generate_all_paper_figures.py` | `table_exp_b_few_shot_adaptation_curve.csv` |
| **Fig. 13** | TreeSHAP Feature Attribution & Interaction Rankings | `scripts/generate_all_paper_figures.py` | `table_exp_c_shap_feature_importance.csv` |
| **Fig. 14** | Attribution Faithfulness Deletion/Insertion Curves | `scripts/plot_paper_figures_09_10_14.py` | `table_exp_c_faithfulness_curves.csv` |
| **Fig. 15** | Single-Query Latency vs. Accuracy Frontier | `scripts/generate_all_paper_figures.py` | `table_exp_e_computational_scalability.csv` |
| **Fig. 16** | Parameter Count vs. F1 Tradeoff & Memory Bubble | `scripts/generate_all_paper_figures.py` | `table_exp_e_computational_scalability.csv` |
| **Fig. 17** | Cumulative Student Effort Time-Budget CDF | `scripts/generate_all_paper_figures.py` | `table_exp_f_time_budget.csv` |
| **Fig. 18** | NLP Latency vs. Taxonomy Correlation Pareto Frontier | `scripts/generate_all_paper_figures.py` | `table_exp_g_transformer_benchmark.csv` |
| **Fig. 19** | Intersectional Disparate Impact Ratio (DIR) Bars | `scripts/generate_all_paper_figures.py` | `table_exp_h_intersectional_fairness.csv` |
| **Fig. 20** | Macro F1 Degradation Under Asymmetric Label Noise | `scripts/generate_all_paper_figures.py` | `table_exp_i_label_noise_imbalance.csv` |
| **Fig. 21** | Macro-Cohort Placement Yield & Institutional ROI Curve | `scripts/generate_all_paper_figures.py` | `table_exp_j_macro_policy_simulation.csv` |

---

## 5. Replication Quickstart Guide

### 5.1 Environment Prerequisites
```bash
# Clone the repository
git clone https://github.com/MIHMahmudEli/edualign.git
cd edualign

# Install minimal dependencies (Python >= 3.9)
pip install -r requirements.txt
```

### 5.2 List All Replication Modules
To display all 18 research modules and their corresponding manuscript tables and figures:
```bash
python scripts/run_all_experiments.py --list
```

### 5.3 Single-Command Full Replication
To execute all 18 experimental modules sequentially and re-generate all canonical CSV matrices in `derived_matrices/`:
```bash
python scripts/run_all_experiments.py --all
```

### 5.4 Selective Module Execution
To execute an individual experimental module:
```bash
python scripts/run_all_experiments.py --module 2   # Deep Stacking Meta-Ensemble benchmark
python scripts/run_all_experiments.py --module 4   # Wilcoxon tests & bootstrap 95% CIs
python scripts/run_all_experiments.py --module 10  # Algorithmic fairness audit
python scripts/run_all_experiments.py --module 15  # Macro-cohort institutional policy simulation
python scripts/run_all_experiments.py --module 18  # Intersectional demographic fairness audit
```

### 5.5 Deterministic Figure Generation
To re-generate all publication-ready vector figures:
```bash
python scripts/generate_all_paper_figures.py
python scripts/plot_paper_figures_09_10_14.py
```

---

## 6. Citation

If you utilize this framework, empirical dataset suite, or baseline models in your research, please cite:

```bibtex
@article{hossain2026bridging,
  title={Bridging the Academic--Industry Skills Divide: A Multi-Source Deep Learning Framework for Dense Semantic Gap Quantification and Prescriptive Employability Intelligence},
  author={Hossain, Mohsin Ibna and Kabir, Humayan and Ahmed, Firoz and Patwary, Md. Owafeeuzzaman},
  journal={Working Paper / Under Review},
  year={2026},
  note={Manuscript submitted for peer review. Full bibliographic metadata will be updated upon formal acceptance. Replication package available at https://github.com/MIHMahmudEli/edualign}
}
```

---

## 7. Data Ethics & Governance

- **Anonymization:** All individual student, employee, and institutional identifier keys have been pseudonymized via non-reversible SHA-256 hashes prior to ingestion.
- **Fairness Compliance:** Audited under IEEE and US EEOC regulatory guidelines, achieving Disparate Impact Ratio ($\text{DIR} \ge 0.9922$) and Equalized Odds Distance ($\text{EOD} \le 0.0021$).
- **Licensing:** Raw datasets are analyzed under their respective Open Data Commons / CC-BY licenses; code is released under the MIT License.
