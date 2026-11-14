"""
Master Experimental Orchestrator & Replication Harness
=====================================================
Automated, deterministic replication pipeline for:
"Bridging the Academic--Industry Skills Divide: A Multi-Source Deep Learning
Framework for Dense Semantic Gap Quantification and Prescriptive Employability Intelligence"

Author: Research Team
License: Open-Access Academic Replication License
"""

import sys
import os
import time
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts") if os.path.basename(BASE_DIR) != 'scripts' else BASE_DIR
DATASETS_DIR = os.path.dirname(SCRIPTS_DIR) if os.path.basename(BASE_DIR) == 'scripts' else BASE_DIR
MATRICES_DIR = os.path.join(DATASETS_DIR, "derived_matrices")

# Structured registry of experimental modules mapped to manuscript RQs
EXPERIMENTS = [
    {
        "module": "1. Baselines & Initial Tests",
        "script": "statistical_tests.py",
        "desc": "Baseline descriptive stats, Chi-square tests, correlation, Logistic Regression, Random Forest",
        "output": ["table1_descriptive_statistics.csv", "table2_chi_square_tests.csv", "table3_correlation_matrix.csv"]
    },
    {
        "module": "2. Core Pipeline & 10-Fold Benchmark",
        "script": "advanced_pipeline.py",
        "desc": "Continuous SVD semantic gap, 10-fold stratified CV, Deep MLP, Stacking meta-ensemble, counterfactuals",
        "output": ["table_semantic_skill_gaps.csv", "table_model_benchmark_10fold.csv", "table_counterfactual_interventions.csv"]
    },
    {
        "module": "3. Diagnostics & ROC Matrices",
        "script": "generate_roc_and_confusion_matrices.py",
        "desc": "ROC thresholds, multi-class confusion matrices, sensitivity/specificity curves",
        "output": ["table_model_roc_curves.csv", "table_confusion_matrices.csv"]
    },
    {
        "module": "4. Statistical Hypothesis Verification",
        "script": "exp_a_statistical_significance.py",
        "desc": "Wilcoxon signed-rank tests, 10,000-fold bootstrap 95% confidence intervals",
        "output": ["table_exp_a_statistical_tests.csv", "table_exp_a_bootstrap_ci.csv"]
    },
    {
        "module": "5. Cross-Cohort Transferability",
        "script": "exp_b_cross_cohort_transfer.py",
        "desc": "Zero-shot transfer across Domestic (10K), Career (50K), and International (300K) cohorts",
        "output": ["table_exp_b_zero_shot_transfer_matrix.csv", "table_exp_b_covariate_shift_metrics.csv"]
    },
    {
        "module": "6. Tree-SHAP Synergy & Faithfulness",
        "script": "exp_c_shap_interaction_faithfulness.py",
        "desc": "Second-order SHAP interaction matrices and feature-degradation attribution faithfulness curves",
        "output": ["table_exp_c_shap_interaction_matrix_abs.csv", "table_exp_c_faithfulness_curves.csv"]
    },
    {
        "module": "7. Compensatory Capital Surface",
        "script": "exp2_compensatory_interaction.py",
        "desc": "2D partial dependence interaction between CGPA and technical projects",
        "output": ["table_exp2_cgpa_projects_pdp_grid.csv", "table_exp2_compensatory_exchange_rate.csv"]
    },
    {
        "module": "8. Pareto Actionability Frontier",
        "script": "exp_d_counterfactual_sparsity_pareto.py",
        "desc": "Constrained recourse optimization, L0 sparsity trade-off, LOF manifold compliance",
        "output": ["table_exp_d_pareto_actionability_frontier.csv", "table_exp_d_counterfactual_sparsity.csv"]
    },
    {
        "module": "9. Computational Scalability SLA",
        "script": "exp_e_computational_scalability.py",
        "desc": "Batch size latency scaling, memory consumption, throughput benchmarking",
        "output": ["table_exp_e_computational_scalability.csv"]
    },
    {
        "module": "10. Algorithmic Governance Audit",
        "script": "exp1_fairness_audit.py",
        "desc": "Intersectional demographic parity (DIR >= 0.992), Equalized Odds (EOD <= 0.002)",
        "output": ["table_exp1_algorithmic_fairness_audit.csv"]
    },
    {
        "module": "11. Longitudinal Skill Velocity",
        "script": "exp3_temporal_velocity.py",
        "desc": "2020-2026 momentum tracking and half-life skill depreciation curves",
        "output": ["table_exp3_temporal_skill_decay.csv", "table_exp3_yearly_penetration_rates.csv"]
    },
    {
        "module": "12. Gaussian Noise Sensitivity",
        "script": "exp4_noise_sensitivity.py",
        "desc": "Empirical stress-testing under injected Gaussian covariate noise",
        "output": ["table_exp4_noise_stress_testing.csv"]
    },
    {
        "module": "13. NLP Representation Ablation",
        "script": "exp5_nlp_ablation.py",
        "desc": "Sub-word char n-grams vs. Sentence-BERT vs. RoBERTa pareto efficiency",
        "output": ["table_exp5_nlp_representation_ablation.csv"]
    },
    {
        "module": "14. Severe Imbalance & Label Noise",
        "script": "run_module4_noise_imbalance.py",
        "desc": "Asymmetric label flip noise stress-testing and minority class recall",
        "output": ["table_exp_i_label_noise_imbalance.csv"]
    },
    {
        "module": "15. Macro-Cohort Policy Simulation",
        "script": "run_module5_policy_simulation.py",
        "desc": "Institutional cost-benefit analysis and dynamic curricular feedback loop (9.8x BCR)",
        "output": ["table_exp_j_macro_policy_simulation.csv"]
    },
    {
        "module": "16. Manifold Plausibility & Time-Budget Feasibility",
        "script": "exp_f_manifold_time_budget.py",
        "desc": "Counterfactual LOF manifold compliance and semestral time-budget audit",
        "output": ["table_exp_f_counterfactual_plausibility.csv", "table_exp_f_manifold_audit.csv", "table_exp_f_time_budget.csv"]
    },
    {
        "module": "17. Cross-Encoder Transformer Benchmark",
        "script": "exp_g_transformer_benchmark.py",
        "desc": "Cross-encoder transformer, static embedding, and char n-gram SVD efficiency benchmark",
        "output": ["table_exp_g_transformer_benchmark.csv"]
    },
    {
        "module": "18. Intersectional Demographic Fairness Audit",
        "script": "exp_h_intersectional_fairness.py",
        "desc": "Intersectional multi-group demographic parity (DIR, EOD, AAOD) across gender and origin",
        "output": ["table_exp_h_intersectional_fairness.csv"]
    }
]


def run_experiment(exp_info, python_exe=sys.executable):
    script_path = os.path.join(SCRIPTS_DIR, exp_info["script"])
    print("\n" + "=" * 80)
    print(f"RUNNING: {exp_info['module']} ({exp_info['script']})")
    print(f"Description: {exp_info['desc']}")
    print("=" * 80)
    
    t0 = time.time()
    res = subprocess.run([python_exe, script_path], cwd=SCRIPTS_DIR, capture_output=False)
    elapsed = time.time() - t0
    
    if res.returncode == 0:
        print(f"--> [SUCCESS] Completed in {elapsed:.2f}s")
        for out in exp_info["output"]:
            out_path = os.path.join(MATRICES_DIR, out)
            status = "FOUND" if os.path.exists(out_path) else "MISSING"
            print(f"    - Output: {out} [{status}]")
        return True
    else:
        print(f"--> [ERROR] Exited with code {res.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Replication Harness for Academic-Industry Skills Paper")
    parser.add_argument("--all", action="store_true", help="Execute all 15 experiment modules sequentially")
    parser.add_argument("--module", type=int, default=None, help="Execute specific module index (1-15)")
    parser.add_argument("--list", action="store_true", help="List all available experiment modules")
    args = parser.parse_args()

    if args.list or (not args.all and args.module is None):
        print("\nAvailable Replication Modules:")
        print("-" * 80)
        for i, exp in enumerate(EXPERIMENTS, start=1):
            print(f"[{i:02d}] {exp['script']:<38} | {exp['module']}")
            print(f"     -> {exp['desc']}")
        print("-" * 80)
        print("Usage: python run_all_experiments.py --all")
        print("       python run_all_experiments.py --module <1-18>")
        return

    os.makedirs(MATRICES_DIR, exist_ok=True)

    if args.module is not None:
        if 1 <= args.module <= len(EXPERIMENTS):
            run_experiment(EXPERIMENTS[args.module - 1])
        else:
            print(f"Invalid module index: {args.module}. Must be between 1 and {len(EXPERIMENTS)}.")
    elif args.all:
        print("\nStarting full replication suite across all 15 modules...")
        success_count = 0
        total_time = time.time()
        for exp in EXPERIMENTS:
            if run_experiment(exp):
                success_count += 1
        elapsed = time.time() - total_time
        print("\n" + "=" * 80)
        print(f"REPLICATION SUITE FINISHED: {success_count}/{len(EXPERIMENTS)} modules succeeded in {elapsed:.1f}s.")
        print(f"All derived matrices are persisted in: {MATRICES_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    main()
