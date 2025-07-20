#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd
from scipy import stats
import os

def calculate_subject_preference_stats(results_df, total_comparisons_per_subject=None):
    """
    Calculate subject preference statistics with confidence intervals.
    
    Args:
        results_df: DataFrame with experiment results
        total_comparisons_per_subject: Optional dict of total comparisons per subject
        
    Returns:
        Dict with subject stats including win rates and confidence intervals
    """
    winners = results_df[results_df['winner_subject'].notna()]
    subject_wins = Counter(winners['winner_subject'])
    
    # Count total appearances for each subject (wins + losses)
    all_subjects_mentioned = list(winners['task_a_subject']) + list(winners['task_b_subject'])
    subject_appearances = Counter(all_subjects_mentioned)
    
    subject_stats = {}
    
    for subject in subject_wins.keys():
        wins = subject_wins[subject]
        total = subject_appearances[subject]
        
        if total > 0:
            win_rate = wins / total
            
            # Calculate Wilson score confidence interval for binomial proportion
            n = total
            p = win_rate
            z = 1.96  # 95% confidence interval
            
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
            
            subject_stats[subject] = {
                'wins': wins,
                'total': total,
                'win_rate': win_rate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'error_bar': margin
            }
    
    return subject_stats

def plot_subject_preferences_with_error_bars(results_df, output_path, title_suffix="", top_n=10):
    """
    Plot subject preferences with error bars showing confidence intervals.
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the plot
        title_suffix: Additional text for plot title
        top_n: Number of top subjects to show
    """
    subject_stats = calculate_subject_preference_stats(results_df)
    
    if not subject_stats:
        print(f"No subject data available for plotting {title_suffix}")
        return
    
    # Sort by win rate, then by total comparisons for tie-breaking
    sorted_subjects = sorted(subject_stats.items(), 
                           key=lambda x: (x[1]['win_rate'], x[1]['total']), 
                           reverse=True)
    
    # Take top N subjects
    top_subjects = sorted_subjects[:top_n]
    
    subjects = [s[0] for s in top_subjects]
    win_rates = [s[1]['win_rate'] for s in top_subjects]
    error_bars = [s[1]['error_bar'] for s in top_subjects]
    total_comparisons = [s[1]['total'] for s in top_subjects]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot with error bars
    y_pos = np.arange(len(subjects))
    bars = ax.barh(y_pos, win_rates, xerr=error_bars, 
                   color='lightgreen', alpha=0.7, capsize=5)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace('_', ' ').title() for s in subjects])
    ax.set_xlabel('Win Rate (with 95% CI)')
    ax.set_title(f'Subject Preferences{title_suffix}\n(Top {len(subjects)} by Win Rate)')
    ax.set_xlim(0, 1)
    
    # Add value labels with sample sizes
    for i, (bar, win_rate, total) in enumerate(zip(bars, win_rates, total_comparisons)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{win_rate:.2f} (n={total})', 
               ha='left', va='center', fontsize=9)
    
    # Add reference line at 0.5 (no preference)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='No Preference (0.5)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    return subject_stats

def plot_position_bias_analysis(results_df, output_path, title_suffix=""):
    """
    Plot position bias analysis focusing only on bias metrics.
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the plot
        title_suffix: Additional text for plot title
    """
    successful = results_df[results_df['success'] == True]
    
    if successful.empty:
        print(f"No successful results for position bias analysis {title_suffix}")
        return
    
    # Calculate overall choice distribution
    choice_counts = successful['choice'].value_counts()
    choice_a_rate = choice_counts.get('A', 0) / len(successful)
    choice_b_rate = 1 - choice_a_rate
    
    # Calculate confidence interval for choice A rate
    n = len(successful)
    p = choice_a_rate
    z = 1.96  # 95% confidence interval
    
    margin = z * np.sqrt(p * (1 - p) / n)
    ci_lower = max(0, p - margin)
    ci_upper = min(1, p + margin)
    
    # Choice by order analysis
    ab_data = successful[successful['order'] == 'AB']
    ba_data = successful[successful['order'] == 'BA']
    
    ab_choice_a = (ab_data['choice'] == 'A').mean() if len(ab_data) > 0 else 0
    ba_choice_a = (ba_data['choice'] == 'A').mean() if len(ba_data) > 0 else 0
    
    # Calculate confidence intervals for order-specific rates
    def calculate_ci(rate, n):
        if n == 0:
            return 0, 0
        margin = z * np.sqrt(rate * (1 - rate) / n)
        return max(0, rate - margin), min(1, rate + margin)
    
    ab_ci_lower, ab_ci_upper = calculate_ci(ab_choice_a, len(ab_data))
    ba_ci_lower, ba_ci_upper = calculate_ci(ba_choice_a, len(ba_data))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall choice distribution with error bars
    choices = ['Choice A', 'Choice B']
    rates = [choice_a_rate, choice_b_rate]
    errors = [margin, margin]  # Same error for both since they sum to 1
    
    bars1 = ax1.bar(choices, rates, yerr=errors, capsize=5, 
                    color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('Choice Rate')
    ax1.set_title(f'Overall Choice Distribution{title_suffix}\n(n={n})')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='No Bias (0.5)')
    ax1.legend()
    
    # Add value labels
    for bar, rate in zip(bars1, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + margin + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Choice by order with error bars
    orders = ['AB Order', 'BA Order']
    choice_a_rates = [ab_choice_a, ba_choice_a]
    order_errors = [ab_choice_a - ab_ci_lower, ba_choice_a - ba_ci_lower]
    
    bars2 = ax2.bar(orders, choice_a_rates, yerr=order_errors, capsize=5,
                    color=['skyblue', 'lightcoral'], alpha=0.7)
    ax2.set_ylabel('Choice A Rate')
    ax2.set_title(f'Position Bias Analysis{title_suffix}\n(AB: n={len(ab_data)}, BA: n={len(ba_data)})')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='No Bias (0.5)')
    ax2.legend()
    
    # Add value labels
    for bar, rate in zip(bars2, choice_a_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    # Calculate and return bias metrics
    position_bias_effect = abs(choice_a_rate - 0.5)
    order_bias_effect = abs(ab_choice_a - ba_choice_a)
    
    return {
        'overall_choice_a_rate': choice_a_rate,
        'choice_a_ci': (ci_lower, ci_upper),
        'ab_choice_a_rate': ab_choice_a,
        'ba_choice_a_rate': ba_choice_a,
        'position_bias_effect': position_bias_effect,
        'order_bias_effect': order_bias_effect,
        'sample_size': n
    }

def analyze_variance_and_sample_adequacy(results_df, subject_stats):
    """
    Analyze variance in results to determine if more samples are needed.
    
    Args:
        results_df: DataFrame with experiment results
        subject_stats: Subject statistics from calculate_subject_preference_stats
        
    Returns:
        Dict with variance analysis and recommendations
    """
    # Calculate variance metrics
    win_rates = [stats['win_rate'] for stats in subject_stats.values()]
    sample_sizes = [stats['total'] for stats in subject_stats.values()]
    error_bars = [stats['error_bar'] for stats in subject_stats.values()]
    
    if not win_rates:
        return {'recommendation': 'insufficient_data', 'reason': 'No subject data available'}
    
    # Calculate overall variance metrics
    win_rate_variance = np.var(win_rates)
    mean_error_bar = np.mean(error_bars)
    min_sample_size = min(sample_sizes)
    max_sample_size = max(sample_sizes)
    
    # Determine if variance is too high relative to signal
    # If mean error bar is > 20% of the range of win rates, we might need more samples
    win_rate_range = max(win_rates) - min(win_rates)
    variance_to_signal_ratio = mean_error_bar / max(win_rate_range, 0.1)  # Avoid division by zero
    
    # Recommendations based on statistical criteria
    recommendations = []
    
    if mean_error_bar > 0.1:  # Error bars > 10 percentage points
        recommendations.append("Large error bars suggest need for more samples")
    
    if variance_to_signal_ratio > 0.5:  # Error bars are > 50% of signal range
        recommendations.append("High variance relative to signal - consider more samples")
    
    if min_sample_size < 10:  # Very small sample sizes
        recommendations.append("Some subjects have very few comparisons")
    
    if max_sample_size / min_sample_size > 5:  # Highly unbalanced
        recommendations.append("Highly unbalanced sample sizes across subjects")
    
    # Overall recommendation
    if len(recommendations) >= 2:
        overall_recommendation = "increase_samples"
    elif len(recommendations) == 1:
        overall_recommendation = "consider_more_samples"
    else:
        overall_recommendation = "samples_adequate"
    
    return {
        'win_rate_variance': win_rate_variance,
        'mean_error_bar': mean_error_bar,
        'variance_to_signal_ratio': variance_to_signal_ratio,
        'min_sample_size': min_sample_size,
        'max_sample_size': max_sample_size,
        'sample_size_ratio': max_sample_size / max(min_sample_size, 1),
        'individual_recommendations': recommendations,
        'overall_recommendation': overall_recommendation,
        'detailed_analysis': {
            'total_subjects': len(subject_stats),
            'win_rate_range': win_rate_range,
            'subjects_with_low_samples': sum(1 for s in sample_sizes if s < 10)
        }
    }

def create_comprehensive_analysis_plots(results_df, output_dir, experiment_name=""):
    """
    Create comprehensive analysis plots focusing on bias and preferences only.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save plots
        experiment_name: Name for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Position Bias Analysis
    position_bias_path = os.path.join(output_dir, 'position_bias_analysis.png')
    bias_stats = plot_position_bias_analysis(results_df, position_bias_path, f" - {experiment_name}")
    
    # 2. Subject Preferences with Error Bars
    subject_pref_path = os.path.join(output_dir, 'subject_preferences.png')
    subject_stats = plot_subject_preferences_with_error_bars(results_df, subject_pref_path, f" - {experiment_name}")
    
    # 3. Variance Analysis
    variance_analysis = analyze_variance_and_sample_adequacy(results_df, subject_stats)
    
    # Create summary plot of variance analysis
    variance_plot_path = os.path.join(output_dir, 'variance_analysis.png')
    create_variance_summary_plot(subject_stats, variance_analysis, variance_plot_path, f" - {experiment_name}")
    
    return {
        'bias_stats': bias_stats,
        'subject_stats': subject_stats,
        'variance_analysis': variance_analysis
    }

def create_variance_summary_plot(subject_stats, variance_analysis, output_path, title_suffix=""):
    """Create a summary plot showing variance analysis results."""
    
    # Extract data for plotting
    subjects = list(subject_stats.keys())[:10]  # Top 10
    win_rates = [subject_stats[s]['win_rate'] for s in subjects]
    error_bars = [subject_stats[s]['error_bar'] for s in subjects]
    sample_sizes = [subject_stats[s]['total'] for s in subjects]
    
    if not subjects:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Win rates with error bars colored by sample size
    colors = plt.cm.viridis([s/max(sample_sizes) for s in sample_sizes])
    
    bars = ax1.bar(range(len(subjects)), win_rates, yerr=error_bars, 
                   color=colors, alpha=0.7, capsize=3)
    ax1.set_xticks(range(len(subjects)))
    ax1.set_xticklabels([s.replace('_', ' ')[:10] for s in subjects], rotation=45, ha='right')
    ax1.set_ylabel('Win Rate')
    ax1.set_title(f'Subject Win Rates with Error Bars{title_suffix}')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='No Preference')
    ax1.legend()
    
    # Add colorbar for sample sizes
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                               norm=plt.Normalize(vmin=min(sample_sizes), vmax=max(sample_sizes)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Sample Size')
    
    # Plot 2: Error bar vs sample size scatter
    ax2.scatter(sample_sizes, error_bars, alpha=0.7, color='blue')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Error Bar Size')
    ax2.set_title(f'Error Bar Size vs Sample Size{title_suffix}')
    
    # Add trend line
    if len(sample_sizes) > 1:
        z = np.polyfit(sample_sizes, error_bars, 1)
        p = np.poly1d(z)
        ax2.plot(sample_sizes, p(sample_sizes), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def main():
    """Test the improved plotting functions."""
    print("Improved plotting module loaded successfully!")
    print("Functions available:")
    print("- plot_subject_preferences_with_error_bars()")
    print("- plot_position_bias_analysis()")
    print("- analyze_variance_and_sample_adequacy()")
    print("- create_comprehensive_analysis_plots()")

if __name__ == "__main__":
    main()