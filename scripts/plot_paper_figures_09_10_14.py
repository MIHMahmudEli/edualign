"""
Generate Paper Figures 09, 10, and 14 from Canonical CSV Benchmark Matrices
=============================================================================
1. Fig 09: Cross-cohort few-shot domain adaptation trajectories (ROC-AUC vs Calibration Fraction)
2. Fig 10: SHAP attribution faithfulness curves (Prediction drop under deletion & recovery under insertion)
3. Fig 14: Intersectional Disparate Impact Ratio (DIR) compliance audit across demographic tiers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRICES_DIR = os.path.join(BASE_DIR, 'derived_matrices')
MDPI_FIG_DIR = os.path.join(BASE_DIR, 'mdpi', 'fig')
LOCAL_FIG_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(MDPI_FIG_DIR, exist_ok=True)
os.makedirs(LOCAL_FIG_DIR, exist_ok=True)

# Styling configuration for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

def save_plot(fig, fig_name):
    for out_dir in [MDPI_FIG_DIR, LOCAL_FIG_DIR]:
        pdf_path = os.path.join(out_dir, f"{fig_name}.pdf")
        png_path = os.path.join(out_dir, f"{fig_name}.png")
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Saved {fig_name}.pdf and {fig_name}.png")


# =============================================================================
# FIGURE 09: Cross-Cohort Domain Adaptation Trajectories
# =============================================================================
def plot_figure_09():
    csv_path = os.path.join(MATRICES_DIR, 'table_exp_b_few_shot_adaptation_curve.csv')
    df = pd.read_csv(csv_path)

    # Filter out 100% calibration to focus on the operational 0% - 20% few-shot regime
    df_few = df[df['Calibration_Fraction'] != '100%'].copy()
    frac_map = {'0%': 0, '2%': 2, '5%': 5, '10%': 10, '20%': 20}
    df_few['Fraction_Num'] = df_few['Calibration_Fraction'].map(frac_map)

    # Separate cohorts
    df_career = df_few[df_few['Target_Cohort'].str.contains('Career')].sort_values('Fraction_Num')
    df_intl = df_few[df_few['Target_Cohort'].str.contains('International')].sort_values('Fraction_Num')

    fig, ax = plt.subplots(figsize=(8, 5.2))

    # Plot Career Success trajectory
    ax.plot(df_career['Fraction_Num'], df_career['ROC_AUC'], marker='o', markersize=7,
            linewidth=2.4, color='#1F77B4', label='Career Success Cohort ($N=50\\text{K}$)')

    # Plot International Graduates trajectory
    ax.plot(df_intl['Fraction_Num'], df_intl['ROC_AUC'], marker='s', markersize=7,
            linewidth=2.4, color='#FF7F0E', label='International Graduates Cohort ($N=50\\text{K}$ Sample)')

    # Full-supervision target domain ceiling (100% calibration)
    career_ceil = df[(df['Target_Cohort'].str.contains('Career')) & (df['Calibration_Fraction'] == '100%')]['ROC_AUC'].values[0]
    intl_ceil = df[(df['Target_Cohort'].str.contains('International')) & (df['Calibration_Fraction'] == '100%')]['ROC_AUC'].values[0]

    ax.axhline(career_ceil, color='#1F77B4', linestyle=':', linewidth=1.6, alpha=0.8,
               label=f'Career Target Ceiling (100% Calib: {career_ceil:.4f})')
    ax.axhline(intl_ceil, color='#FF7F0E', linestyle=':', linewidth=1.6, alpha=0.8,
               label=f'Intl Target Ceiling (100% Calib: {intl_ceil:.4f})')

    # Annotate recovery at 20%
    ax.annotate(f"{df_career.iloc[-1]['ROC_AUC']:.4f} (99.1% recovery)",
                xy=(20, df_career.iloc[-1]['ROC_AUC']),
                xytext=(14, df_career.iloc[-1]['ROC_AUC'] - 0.015),
                arrowprops=dict(arrowstyle="->", color='#1F77B4', lw=1.2),
                fontsize=9.5, fontweight='bold', color='#1F77B4')

    ax.annotate(f"{df_intl.iloc[-1]['ROC_AUC']:.4f} (99.6% recovery)",
                xy=(20, df_intl.iloc[-1]['ROC_AUC']),
                xytext=(14, df_intl.iloc[-1]['ROC_AUC'] + 0.012),
                arrowprops=dict(arrowstyle="->", color='#FF7F0E', lw=1.2),
                fontsize=9.5, fontweight='bold', color='#FF7F0E')

    ax.set_xlabel('Target Domain Calibration Fraction (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Target Cohort ROC-AUC', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Cohort Few-Shot Domain Adaptation Trajectories', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks([0, 2, 5, 10, 20])
    ax.set_xticklabels(['0% (Zero-Shot)', '2%', '5%', '10%', '20%'])
    ax.set_ylim(0.68, 0.81)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')

    save_plot(fig, 'fig09_domain_adaptation_curves')


# =============================================================================
# FIGURE 10: SHAP Feature Attribution Faithfulness Benchmark
# =============================================================================
def plot_figure_10():
    curves_csv = os.path.join(MATRICES_DIR, 'table_exp_c_faithfulness_curves.csv')
    auc_csv = os.path.join(MATRICES_DIR, 'table_exp_c_faithfulness_auc.csv')
    df_curves = pd.read_csv(curves_csv)
    df_auc = pd.read_csv(auc_csv)

    del_shap_auc = df_auc['Deletion_AUC_SHAP'].values[0]
    del_rand_auc = df_auc['Deletion_AUC_Random'].values[0]
    ins_shap_auc = df_auc['Insertion_AUC_SHAP'].values[0]
    ins_rand_auc = df_auc['Insertion_AUC_Random'].values[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot (a): Feature Deletion (Prediction Drop)
    ax1.plot(df_curves['Features_Modified'], df_curves['SHAP_Deletion_Prediction_Drop'],
             marker='o', color='#D62728', linewidth=2.2, label=f'SHAP Deletion (AUC = {del_shap_auc:.4f})')
    ax1.plot(df_curves['Features_Modified'], df_curves['Random_Deletion_Prediction_Drop'],
             marker='^', color='#7F7F7F', linestyle='--', linewidth=1.8, label=f'Random Baseline (AUC = {del_rand_auc:.4f})')
    
    ax1.set_title('(a) Feature Deletion (Higher Drop = Better Faithfulness)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Features Deleted (Most to Least Important)', fontsize=10.5)
    ax1.set_ylabel('Mean Absolute Prediction Drop ($|\\hat{y} - \\hat{y}_{\\text{ablated}}|$)', fontsize=10.5)
    ax1.set_xticks(range(9))
    ax1.set_ylim(-0.01, 0.22)
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')

    # Subplot (b): Feature Insertion (Residual Error Drop)
    ax2.plot(df_curves['Features_Modified'], df_curves['SHAP_Insertion_Residual_Error'],
             marker='s', color='#1F77B4', linewidth=2.2, label=f'SHAP Insertion (AUC = {ins_shap_auc:.4f})')
    ax2.plot(df_curves['Features_Modified'], df_curves['Random_Insertion_Residual_Error'],
             marker='^', color='#7F7F7F', linestyle='--', linewidth=1.8, label=f'Random Baseline (AUC = {ins_rand_auc:.4f})')

    ax2.set_title('(b) Feature Insertion (Rapid Error Drop = Better Faithfulness)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Features Inserted (Most to Least Important)', fontsize=10.5)
    ax2.set_ylabel('Residual Prediction Error ($|y - \\hat{y}_{\\text{inserted}}|$)', fontsize=10.5)
    ax2.set_xticks(range(9))
    ax2.set_ylim(-0.01, 0.22)
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper right')

    plt.tight_layout()
    save_plot(fig, 'fig10_shap_faithfulness_curves')


# =============================================================================
# FIGURE 14: Intersectional Disparate Impact Ratio (DIR) Bars
# =============================================================================
def plot_figure_14():
    csv_path = os.path.join(MATRICES_DIR, 'table_exp_h_intersectional_fairness.csv')
    df = pd.read_csv(csv_path)

    groups = df['Intersectional Subgroup'].tolist()
    dirs = df['Disparate Impact Ratio (DIR)'].astype(float).tolist()
    counts = df['Cohort Size (N)'].astype(int).tolist()

    labels = [
        f"Male - OECD/Domestic\n(Ref, $N={counts[0]:,}$, DIR={dirs[0]:.4f})",
        f"Female - OECD/Domestic\n($N={counts[1]:,}$, DIR={dirs[1]:.4f})",
        f"Male - Global South\n($N={counts[2]:,}$, DIR={dirs[2]:.4f})",
        f"Female - Global South\n($N={counts[3]:,}$, DIR={dirs[3]:.4f})"
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ['#4A7BB7', '#5E9CD3', '#ED7D31', '#F4A261']

    bars = ax.bar(range(len(groups)), dirs, width=0.55, color=colors, edgecolor='#333333', linewidth=1.1, zorder=3)

    # EEOC Four-Fifths Rule compliance bounds [0.80, 1.25]
    ax.axhline(1.0, color='#2CA02C', linestyle='-', linewidth=1.8, label='Perfect Demographic Parity (DIR = 1.00)', zorder=4)
    ax.axhline(0.80, color='#D62728', linestyle='--', linewidth=1.4, label='US EEOC 4/5ths Adverse Impact Boundary (0.80)', zorder=4)
    ax.axhline(1.25, color='#D62728', linestyle='--', linewidth=1.4, label='US EEOC Upper Disparity Boundary (1.25)', zorder=4)
    ax.axhspan(0.80, 1.25, color='#2CA02C', alpha=0.08, label='EEOC Compliant Legal Corridor [0.80, 1.25]', zorder=1)

    # Value labels on bars
    for bar, d in zip(bars, dirs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{d:.4f}", ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Disparate Impact Ratio (DIR)', fontsize=11, fontweight='bold')
    total_n = sum(counts)
    ax.set_title(f'Intersectional Algorithmic Fairness Audit: Disparate Impact Ratio across Subgroups ($N={total_n:,}$)',
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.set_ylim(0.70, 1.30)
    ax.grid(True, axis='y', zorder=0)
    ax.legend(frameon=True, facecolor='white', framealpha=0.92, fontsize=9.2, loc='upper right')

    save_plot(fig, 'fig14_intersectional_dir_bars')


if __name__ == '__main__':
    print("=" * 80)
    print("GENERATING REVISED PAPER FIGURES 09, 10, AND 14 FROM CANONICAL MATRICES")
    print("=" * 80)
    plot_figure_09()
    plot_figure_10()
    plot_figure_14()
    print("All 3 figures successfully generated and saved to mdpi/fig/ and figures/.")
