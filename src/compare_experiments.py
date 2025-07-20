#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/dcruz/model_preferences/src')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from scipy import stats
import json

def compare_experiment_types():
    """Compare results between multiple choice and open-ended experiments."""
    
    # Paths to experiment results
    mc_path = "/home/dcruz/model_preferences/experiments/20250719_232812_focused_representative"
    oe_path = "/home/dcruz/model_preferences/experiments/20250720_002629_open_ended_comparison_20tasks"
    
    # Load results
    mc_results = pd.read_csv(os.path.join(mc_path, "results.csv"))
    oe_results = pd.read_csv(os.path.join(oe_path, "results.csv"))
    
    # Load metadata
    with open(os.path.join(mc_path, "metadata.json")) as f:
        mc_metadata = json.load(f)
    
    with open(os.path.join(oe_path, "metadata.json")) as f:
        oe_metadata = json.load(f)
    
    print("=" * 80)
    print("MULTIPLE CHOICE vs OPEN-ENDED COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Basic comparison
    print("\n1. EXPERIMENT OVERVIEW")
    print("-" * 50)
    print(f"Multiple Choice (MC) Experiment:")
    print(f"  - Total comparisons: {len(mc_results)}")
    print(f"  - Successful: {mc_results['success'].sum()} ({mc_results['success'].mean():.1%})")
    print(f"  - Model: {mc_metadata['model_name']}")
    print(f"  - Tasks: {mc_metadata['n_tasks']}")
    
    print(f"\nOpen-Ended (OE) Experiment:")
    print(f"  - Total comparisons: {len(oe_results)}")
    print(f"  - Successful: {oe_results['success'].sum()} ({oe_results['success'].mean():.1%})")
    print(f"  - Model: {oe_metadata['model_name']}")
    print(f"  - Tasks: {oe_metadata['n_tasks']}")
    
    # Position bias comparison
    print("\n2. POSITION BIAS ANALYSIS")
    print("-" * 50)
    
    # Filter successful results
    mc_successful = mc_results[mc_results['success'] == True]
    oe_successful = oe_results[oe_results['success'] == True]
    
    # Calculate choice A rates
    mc_choice_a_rate = (mc_successful['choice'] == 'A').mean()
    oe_choice_a_rate = (oe_successful['choice'] == 'A').mean()
    
    print(f"Multiple Choice - Choice A rate: {mc_choice_a_rate:.1%}")
    print(f"Open-Ended - Choice A rate: {oe_choice_a_rate:.1%}")
    print(f"Difference: {abs(mc_choice_a_rate - oe_choice_a_rate):.1%}")
    
    # Statistical significance test
    mc_choice_a_count = (mc_successful['choice'] == 'A').sum()
    oe_choice_a_count = (oe_successful['choice'] == 'A').sum()
    
    # Chi-square test for independence
    from scipy.stats import chi2_contingency
    
    contingency_table = [
        [mc_choice_a_count, len(mc_successful) - mc_choice_a_count],
        [oe_choice_a_count, len(oe_successful) - oe_choice_a_count]
    ]
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Statistical test (Chi-square): p = {p_value:.6f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Order bias breakdown
    print(f"\n3. ORDER BIAS BREAKDOWN")
    print("-" * 50)
    
    for experiment, data, name in [(mc_successful, mc_successful, "Multiple Choice"), 
                                   (oe_successful, oe_successful, "Open-Ended")]:
        ab_data = data[data['order'] == 'AB']
        ba_data = data[data['order'] == 'BA']
        
        ab_choice_a = (ab_data['choice'] == 'A').mean() if len(ab_data) > 0 else 0
        ba_choice_a = (ba_data['choice'] == 'A').mean() if len(ba_data) > 0 else 0
        
        print(f"{name}:")
        print(f"  AB order - Choice A: {ab_choice_a:.1%}")
        print(f"  BA order - Choice A: {ba_choice_a:.1%}")
        print(f"  Order bias difference: {abs(ab_choice_a - ba_choice_a):.1%}")
    
    # Subject preference comparison
    print(f"\n4. SUBJECT PREFERENCE COMPARISON")
    print("-" * 50)
    
    mc_winners = mc_successful[mc_successful['winner_subject'].notna()]
    oe_winners = oe_successful[oe_successful['winner_subject'].notna()]
    
    mc_subject_wins = Counter(mc_winners['winner_subject'])
    oe_subject_wins = Counter(oe_winners['winner_subject'])
    
    print(f"Multiple Choice - Top 5 subjects:")
    for i, (subject, wins) in enumerate(mc_subject_wins.most_common(5), 1):
        win_rate = wins / len(mc_winners)
        print(f"  {i}. {subject.replace('_', ' ').title()}: {wins} wins ({win_rate:.1%})")
    
    print(f"\nOpen-Ended - Top 5 subjects:")
    for i, (subject, wins) in enumerate(oe_subject_wins.most_common(5), 1):
        win_rate = wins / len(oe_winners)
        print(f"  {i}. {subject.replace('_', ' ').title()}: {wins} wins ({win_rate:.1%})")
    
    # Response time comparison
    print(f"\n5. RESPONSE TIME COMPARISON")
    print("-" * 50)
    
    mc_times = mc_successful['response_time'].dropna()
    oe_times = oe_successful['response_time'].dropna()
    
    print(f"Multiple Choice:")
    print(f"  Mean: {mc_times.mean():.3f}s")
    print(f"  Median: {mc_times.median():.3f}s")
    print(f"  Std: {mc_times.std():.3f}s")
    
    print(f"Open-Ended:")
    print(f"  Mean: {oe_times.mean():.3f}s")
    print(f"  Median: {oe_times.median():.3f}s")
    print(f"  Std: {oe_times.std():.3f}s")
    
    # Statistical test for response times
    from scipy.stats import mannwhitneyu
    statistic, p_time = mannwhitneyu(mc_times, oe_times, alternative='two-sided')
    print(f"Response time difference test: p = {p_time:.6f}")
    print(f"Significant difference: {'Yes' if p_time < 0.05 else 'No'}")
    
    # Create comparison plots and save metadata
    output_dir = create_comparison_plots(mc_successful, oe_successful, mc_times, oe_times)
    
    # Save comparison metadata
    from datetime import datetime
    comparison_metadata = {
        'experiment_name': 'format_comparison_analysis',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'experiment_type': 'format_comparison',
        'mc_experiment_path': mc_path,
        'oe_experiment_path': oe_path,
        'mc_total_comparisons': len(mc_results),
        'oe_total_comparisons': len(oe_results),
        'mc_successful_comparisons': len(mc_successful),
        'oe_successful_comparisons': len(oe_successful),
        'mc_choice_a_rate': mc_choice_a_rate,
        'oe_choice_a_rate': oe_choice_a_rate,
        'position_bias_reduction': mc_choice_a_rate - oe_choice_a_rate,
        'statistical_significance_p': p_value,
        'response_time_difference_p': p_time,
        'model_name': mc_metadata['model_name'],
        'seed_used': mc_metadata['seed'],
        'tasks_compared': mc_metadata['n_tasks']
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(comparison_metadata, f, indent=2)
    
    print(f"Comparison analysis saved to: {output_dir}")
    print(f"Metadata saved: {metadata_path}")
    
    return {
        'mc_choice_a_rate': mc_choice_a_rate,
        'oe_choice_a_rate': oe_choice_a_rate,
        'position_bias_p_value': p_value,
        'response_time_p_value': p_time,
        'mc_subject_wins': dict(mc_subject_wins),
        'oe_subject_wins': dict(oe_subject_wins)
    }

def create_comparison_plots(mc_data, oe_data, mc_times, oe_times):
    """Create comprehensive comparison plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create proper timestamped experiment directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/dcruz/model_preferences/experiments/{timestamp}_format_comparison_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Position Bias Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multiple Choice vs Open-Ended Comparison', fontsize=16, fontweight='bold')
    
    # Overall choice rates
    mc_choice_a_rate = (mc_data['choice'] == 'A').mean()
    oe_choice_a_rate = (oe_data['choice'] == 'A').mean()
    
    experiments = ['Multiple Choice', 'Open-Ended']
    choice_a_rates = [mc_choice_a_rate, oe_choice_a_rate]
    
    bars1 = ax1.bar(experiments, choice_a_rates, color=['lightblue', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('Choice A Rate')
    ax1.set_title('Overall Position Bias Comparison')
    ax1.set_ylim(0, 1)
    
    for bar, rate in zip(bars1, choice_a_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Order bias comparison
    mc_ab = mc_data[mc_data['order'] == 'AB']
    mc_ba = mc_data[mc_data['order'] == 'BA']
    oe_ab = oe_data[oe_data['order'] == 'AB']
    oe_ba = oe_data[oe_data['order'] == 'BA']
    
    mc_ab_rate = (mc_ab['choice'] == 'A').mean() if len(mc_ab) > 0 else 0
    mc_ba_rate = (mc_ba['choice'] == 'A').mean() if len(mc_ba) > 0 else 0
    oe_ab_rate = (oe_ab['choice'] == 'A').mean() if len(oe_ab) > 0 else 0
    oe_ba_rate = (oe_ba['choice'] == 'A').mean() if len(oe_ba) > 0 else 0
    
    orders = ['AB Order', 'BA Order']
    mc_rates = [mc_ab_rate, mc_ba_rate]
    oe_rates = [oe_ab_rate, oe_ba_rate]
    
    x = np.arange(len(orders))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, mc_rates, width, label='Multiple Choice', color='lightblue', alpha=0.7)
    bars3 = ax2.bar(x + width/2, oe_rates, width, label='Open-Ended', color='lightcoral', alpha=0.7)
    
    ax2.set_ylabel('Choice A Rate')
    ax2.set_title('Order Bias by Experiment Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(orders)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Response time comparison
    ax3.hist(mc_times, bins=15, alpha=0.6, label='Multiple Choice', color='lightblue', density=True)
    ax3.hist(oe_times, bins=15, alpha=0.6, label='Open-Ended', color='lightcoral', density=True)
    ax3.set_xlabel('Response Time (seconds)')
    ax3.set_ylabel('Density')
    ax3.set_title('Response Time Distribution Comparison')
    ax3.legend()
    
    # Subject preferences comparison (top 5 overlap)
    mc_winners = mc_data[mc_data['winner_subject'].notna()]
    oe_winners = oe_data[oe_data['winner_subject'].notna()]
    
    mc_subject_wins = Counter(mc_winners['winner_subject'])
    oe_subject_wins = Counter(oe_winners['winner_subject'])
    
    # Get top subjects from both experiments
    all_subjects = set(list(mc_subject_wins.keys()) + list(oe_subject_wins.keys()))
    top_subjects = sorted(all_subjects, key=lambda x: mc_subject_wins.get(x, 0) + oe_subject_wins.get(x, 0), reverse=True)[:8]
    
    mc_counts = [mc_subject_wins.get(s, 0) for s in top_subjects]
    oe_counts = [oe_subject_wins.get(s, 0) for s in top_subjects]
    
    x = np.arange(len(top_subjects))
    
    bars4 = ax4.bar(x - width/2, mc_counts, width, label='Multiple Choice', color='lightblue', alpha=0.7)
    bars5 = ax4.bar(x + width/2, oe_counts, width, label='Open-Ended', color='lightcoral', alpha=0.7)
    
    ax4.set_ylabel('Number of Wins')
    ax4.set_title('Subject Preferences Comparison (Top 8)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', ' ')[:10] for s in top_subjects], rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'experiment_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'experiment_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Position Bias Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Position bias effect size
    mc_bias_effect = abs(mc_ab_rate - 0.5)
    oe_bias_effect = abs(oe_ab_rate - 0.5)
    
    experiments = ['Multiple Choice', 'Open-Ended']
    bias_effects = [mc_bias_effect, oe_bias_effect]
    
    bars = ax1.bar(experiments, bias_effects, color=['steelblue', 'indianred'], alpha=0.7)
    ax1.set_ylabel('Position Bias Effect Size')
    ax1.set_title('Position Bias Strength Comparison')
    ax1.set_ylim(0, 0.5)
    
    for bar, effect in zip(bars, bias_effects):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{effect:.2f}', ha='center', va='bottom')
    
    # Response time vs choice pattern
    mc_median_time = mc_times.median()
    oe_median_time = oe_times.median()
    
    mc_fast = mc_data[mc_data['response_time'] <= mc_median_time]
    mc_slow = mc_data[mc_data['response_time'] > mc_median_time]
    oe_fast = oe_data[oe_data['response_time'] <= oe_median_time]
    oe_slow = oe_data[oe_data['response_time'] > oe_median_time]
    
    mc_fast_a = (mc_fast['choice'] == 'A').mean() if len(mc_fast) > 0 else 0
    mc_slow_a = (mc_slow['choice'] == 'A').mean() if len(mc_slow) > 0 else 0
    oe_fast_a = (oe_fast['choice'] == 'A').mean() if len(oe_fast) > 0 else 0
    oe_slow_a = (oe_slow['choice'] == 'A').mean() if len(oe_slow) > 0 else 0
    
    speed_categories = ['Fast\n(≤ median)', 'Slow\n(> median)']
    mc_speed_rates = [mc_fast_a, mc_slow_a]
    oe_speed_rates = [oe_fast_a, oe_slow_a]
    
    x = np.arange(len(speed_categories))
    
    bars1 = ax2.bar(x - width/2, mc_speed_rates, width, label='Multiple Choice', color='lightblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, oe_speed_rates, width, label='Open-Ended', color='lightcoral', alpha=0.7)
    
    ax2.set_ylabel('Choice A Rate')
    ax2.set_title('Response Speed vs Choice Pattern')
    ax2.set_xticks(x)
    ax2.set_xticklabels(speed_categories)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'detailed_bias_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'detailed_bias_analysis.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plots saved to: {plots_dir}")
    
    return output_dir

def main():
    """Run the comparison analysis."""
    results = compare_experiment_types()
    
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)
    print(f"1. Position bias is {'REDUCED' if results['oe_choice_a_rate'] < results['mc_choice_a_rate'] else 'MAINTAINED'} in open-ended format")
    print(f"   Multiple Choice: {results['mc_choice_a_rate']:.1%} → Open-Ended: {results['oe_choice_a_rate']:.1%}")
    print(f"2. Statistical significance: p = {results['position_bias_p_value']:.6f}")
    print(f"3. Open-ended format {'significantly' if results['response_time_p_value'] < 0.05 else 'does not significantly'} change response times")
    
    # Calculate preference overlap
    mc_top_subjects = set(list(results['mc_subject_wins'].keys())[:5])
    oe_top_subjects = set(list(results['oe_subject_wins'].keys())[:5])
    overlap = len(mc_top_subjects.intersection(oe_top_subjects))
    
    print(f"4. Subject preference overlap: {overlap}/5 subjects in both top 5 lists")
    
    return results

if __name__ == "__main__":
    main()