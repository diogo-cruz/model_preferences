#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/dcruz/model_preferences/src')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import json

def analyze_preference_consistency(results_file, output_dir):
    """Analyze how consistent the model is in its preferences across different comparisons."""
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Filter successful comparisons
    successful = df[df['success'] == True].copy()
    
    print("=" * 60)
    print("MODEL PREFERENCE CONSISTENCY ANALYSIS")
    print("=" * 60)
    
    consistency_results = {}
    
    # 1. PAIRWISE CONSISTENCY ANALYSIS
    print("\n1. PAIRWISE CONSISTENCY ANALYSIS")
    print("-" * 40)
    
    # Find direct pairwise comparisons (A vs B and B vs A)
    pairwise_consistency = analyze_pairwise_consistency(successful)
    consistency_results['pairwise_consistency'] = pairwise_consistency
    
    # 2. TRANSITIVITY ANALYSIS  
    print("\n2. TRANSITIVITY ANALYSIS")
    print("-" * 40)
    
    transitivity_results = analyze_transitivity(successful)
    consistency_results['transitivity'] = transitivity_results
    
    # 3. SUBJECT-LEVEL PREFERENCE STABILITY
    print("\n3. SUBJECT-LEVEL PREFERENCE STABILITY")
    print("-" * 40)
    
    subject_stability = analyze_subject_preference_stability(successful)
    consistency_results['subject_stability'] = subject_stability
    
    # 4. ORDER BIAS CONSISTENCY
    print("\n4. ORDER BIAS CONSISTENCY")
    print("-" * 40)
    
    order_consistency = analyze_order_bias_consistency(successful)
    consistency_results['order_consistency'] = order_consistency
    
    # 5. RESPONSE TIME vs CONSISTENCY
    print("\n5. RESPONSE TIME vs CONSISTENCY")
    print("-" * 40)
    
    time_consistency = analyze_response_time_patterns(successful)
    consistency_results['time_patterns'] = time_consistency
    
    # Generate consistency plots
    create_consistency_plots(successful, consistency_results, output_dir)
    
    # Save analysis results
    results_path = os.path.join(output_dir, 'consistency_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(consistency_results, f, indent=2, default=str)
    
    return consistency_results

def analyze_pairwise_consistency(df):
    """Analyze consistency in direct pairwise comparisons (A vs B, B vs A)."""
    
    pairwise_comparisons = defaultdict(list)
    
    # Group comparisons by task pairs
    for _, row in df.iterrows():
        task_pair = tuple(sorted([row['task_a_id'], row['task_b_id']]))
        pairwise_comparisons[task_pair].append({
            'order': row['order'],
            'choice': row['choice'],
            'winner_id': row['winner_id'],
            'task_a_id': row['task_a_id'],
            'task_b_id': row['task_b_id']
        })
    
    consistent_pairs = 0
    inconsistent_pairs = 0
    total_reversible_pairs = 0
    
    consistency_details = []
    
    for task_pair, comparisons in pairwise_comparisons.items():
        if len(comparisons) >= 2:  # Need at least 2 comparisons to check consistency
            total_reversible_pairs += 1
            
            # Find AB and BA comparisons
            ab_comps = [c for c in comparisons if c['order'] == 'AB']
            ba_comps = [c for c in comparisons if c['order'] == 'BA']
            
            if ab_comps and ba_comps:
                # Check if the same task wins in both orders
                ab_winner = ab_comps[0]['winner_id']
                ba_winner = ba_comps[0]['winner_id']
                
                if ab_winner == ba_winner:
                    consistent_pairs += 1
                    is_consistent = True
                else:
                    inconsistent_pairs += 1
                    is_consistent = False
                
                consistency_details.append({
                    'task_pair': task_pair,
                    'ab_winner': ab_winner,
                    'ba_winner': ba_winner,
                    'consistent': is_consistent
                })
    
    consistency_rate = consistent_pairs / total_reversible_pairs if total_reversible_pairs > 0 else 0
    
    print(f"Reversible task pairs found: {total_reversible_pairs}")
    print(f"Consistent pairs: {consistent_pairs}")
    print(f"Inconsistent pairs: {inconsistent_pairs}")
    print(f"Pairwise consistency rate: {consistency_rate:.1%}")
    
    return {
        'total_reversible_pairs': total_reversible_pairs,
        'consistent_pairs': consistent_pairs,
        'inconsistent_pairs': inconsistent_pairs,
        'consistency_rate': consistency_rate,
        'details': consistency_details
    }

def analyze_transitivity(df):
    """Analyze transitivity violations (A > B, B > C, C > A)."""
    
    # Build preference graph
    preferences = {}
    for _, row in df.iterrows():
        winner = row['winner_id']
        loser = row['task_a_id'] if winner == row['task_b_id'] else row['task_b_id']
        
        if winner not in preferences:
            preferences[winner] = set()
        preferences[winner].add(loser)
    
    # Find all tasks
    all_tasks = set(df['task_a_id'].tolist() + df['task_b_id'].tolist())
    
    # Check for transitivity violations
    violations = []
    total_triplets = 0
    
    for a, b, c in combinations(all_tasks, 3):
        if a in preferences and b in preferences.get(a, set()) and \
           b in preferences and c in preferences.get(b, set()) and \
           c in preferences and a in preferences.get(c, set()):
            violations.append((a, b, c))
        
        total_triplets += 1
    
    violation_rate = len(violations) / total_triplets if total_triplets > 0 else 0
    
    print(f"Total possible triplets: {total_triplets}")
    print(f"Transitivity violations found: {len(violations)}")
    print(f"Violation rate: {violation_rate:.3%}")
    
    if violations:
        print("Example violations:")
        for i, (a, b, c) in enumerate(violations[:3]):
            print(f"  {i+1}. Task {a} > Task {b} > Task {c} > Task {a}")
    
    return {
        'total_triplets': total_triplets,
        'violations': len(violations),
        'violation_rate': violation_rate,
        'examples': violations[:5]
    }

def analyze_subject_preference_stability(df):
    """Analyze how stable subject preferences are across different task instances."""
    
    # Group by subject pairs
    subject_pairs = defaultdict(list)
    
    for _, row in df.iterrows():
        subject_pair = tuple(sorted([row['task_a_subject'], row['task_b_subject']]))
        if subject_pair[0] != subject_pair[1]:  # Exclude same-subject comparisons
            subject_pairs[subject_pair].append(row['winner_subject'])
    
    # Analyze consistency within each subject pair
    consistent_subject_pairs = 0
    total_multi_comparison_pairs = 0
    
    subject_stability_details = []
    
    for subject_pair, winners in subject_pairs.items():
        if len(winners) > 1:  # Multiple comparisons of this subject pair
            total_multi_comparison_pairs += 1
            winner_counts = Counter(winners)
            most_common_winner, most_common_count = winner_counts.most_common(1)[0]
            
            consistency_rate = most_common_count / len(winners)
            
            if consistency_rate >= 0.7:  # Consider 70%+ as consistent
                consistent_subject_pairs += 1
            
            subject_stability_details.append({
                'subject_pair': subject_pair,
                'total_comparisons': len(winners),
                'winner_distribution': dict(winner_counts),
                'dominant_winner': most_common_winner,
                'consistency_rate': consistency_rate
            })
    
    overall_stability = consistent_subject_pairs / total_multi_comparison_pairs if total_multi_comparison_pairs > 0 else 0
    
    print(f"Subject pairs with multiple comparisons: {total_multi_comparison_pairs}")
    print(f"Consistent subject pairs (≥70% same winner): {consistent_subject_pairs}")
    print(f"Subject-level stability rate: {overall_stability:.1%}")
    
    return {
        'total_multi_comparison_pairs': total_multi_comparison_pairs,
        'consistent_pairs': consistent_subject_pairs,
        'stability_rate': overall_stability,
        'details': subject_stability_details
    }

def analyze_order_bias_consistency(df):
    """Analyze how consistent the order bias is across different comparisons."""
    
    ab_choices = df[df['order'] == 'AB']['choice'].tolist()
    ba_choices = df[df['order'] == 'BA']['choice'].tolist()
    
    ab_choice_a_rate = ab_choices.count('A') / len(ab_choices) if ab_choices else 0
    ba_choice_a_rate = ba_choices.count('A') / len(ba_choices) if ba_choices else 0
    
    # Calculate consistency of position bias
    bias_difference = abs(ab_choice_a_rate - ba_choice_a_rate)
    bias_consistency = 1 - bias_difference  # High consistency = low difference
    
    print(f"AB order - Choice A rate: {ab_choice_a_rate:.1%}")
    print(f"BA order - Choice A rate: {ba_choice_a_rate:.1%}")
    print(f"Position bias difference: {bias_difference:.1%}")
    print(f"Position bias consistency: {bias_consistency:.1%}")
    
    return {
        'ab_choice_a_rate': ab_choice_a_rate,
        'ba_choice_a_rate': ba_choice_a_rate,
        'bias_difference': bias_difference,
        'bias_consistency': bias_consistency
    }

def analyze_response_time_patterns(df):
    """Analyze relationship between response time and decision consistency."""
    
    # Analyze response time distribution
    response_times = df['response_time'].dropna()
    
    time_stats = {
        'mean': response_times.mean(),
        'median': response_times.median(),
        'std': response_times.std(),
        'min': response_times.min(),
        'max': response_times.max()
    }
    
    # Check if faster responses correlate with position bias
    median_time = response_times.median()
    fast_responses = df[df['response_time'] <= median_time]
    slow_responses = df[df['response_time'] > median_time]
    
    fast_choice_a_rate = (fast_responses['choice'] == 'A').mean()
    slow_choice_a_rate = (slow_responses['choice'] == 'A').mean()
    
    time_bias_difference = abs(fast_choice_a_rate - slow_choice_a_rate)
    
    print(f"Mean response time: {time_stats['mean']:.3f}s")
    print(f"Fast responses (<= median) Choice A rate: {fast_choice_a_rate:.1%}")
    print(f"Slow responses (> median) Choice A rate: {slow_choice_a_rate:.1%}")
    print(f"Time-based bias difference: {time_bias_difference:.1%}")
    
    return {
        'time_stats': time_stats,
        'fast_choice_a_rate': fast_choice_a_rate,
        'slow_choice_a_rate': slow_choice_a_rate,
        'time_bias_difference': time_bias_difference
    }

def create_consistency_plots(df, consistency_results, output_dir):
    """Create comprehensive consistency visualization plots."""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Consistency Summary Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Preference Consistency Analysis', fontsize=16, fontweight='bold')
    
    # Pairwise consistency
    pairwise = consistency_results['pairwise_consistency']
    ax1.pie([pairwise['consistent_pairs'], pairwise['inconsistent_pairs']], 
            labels=['Consistent', 'Inconsistent'], autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Pairwise Consistency\n({pairwise["consistency_rate"]:.1%} overall)')
    
    # Transitivity violations
    transitivity = consistency_results['transitivity']
    violation_rate = transitivity['violation_rate'] * 100
    ax2.bar(['Transitive', 'Violations'], 
            [100 - violation_rate, violation_rate],
            color=['green', 'red'], alpha=0.7)
    ax2.set_title(f'Transitivity Analysis\n({transitivity["violation_rate"]:.3%} violations)')
    ax2.set_ylabel('Percentage')
    
    # Subject stability
    stability = consistency_results['subject_stability']
    ax3.pie([stability['consistent_pairs'], 
             stability['total_multi_comparison_pairs'] - stability['consistent_pairs']], 
            labels=['Stable (≥70%)', 'Unstable'], autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Subject Preference Stability\n({stability["stability_rate"]:.1%} stable)')
    
    # Order bias consistency
    order_bias = consistency_results['order_consistency']
    ax4.bar(['AB Order', 'BA Order'], 
            [order_bias['ab_choice_a_rate'], order_bias['ba_choice_a_rate']],
            color=['blue', 'orange'], alpha=0.7)
    ax4.set_title(f'Order Bias Consistency\n({order_bias["bias_difference"]:.1%} difference)')
    ax4.set_ylabel('Choice A Rate')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'consistency_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'consistency_dashboard.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Response Time vs Choice Pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Response time distribution
    response_times = df['response_time'].dropna()
    ax1.hist(response_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(response_times.mean(), color='red', linestyle='--', 
                label=f'Mean: {response_times.mean():.3f}s')
    ax1.axvline(response_times.median(), color='orange', linestyle='--', 
                label=f'Median: {response_times.median():.3f}s')
    ax1.set_xlabel('Response Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Response Time Distribution')
    ax1.legend()
    
    # Response time vs choice pattern
    time_patterns = consistency_results['time_patterns']
    median_time = response_times.median()
    
    categories = ['Fast\n(≤ median)', 'Slow\n(> median)']
    choice_a_rates = [time_patterns['fast_choice_a_rate'], time_patterns['slow_choice_a_rate']]
    
    bars = ax2.bar(categories, choice_a_rates, color=['lightcoral', 'lightsteelblue'], alpha=0.7)
    ax2.set_ylabel('Choice A Rate')
    ax2.set_title('Response Speed vs Choice Pattern')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, choice_a_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'response_time_patterns.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'response_time_patterns.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Subject Preference Stability Details
    if consistency_results['subject_stability']['details']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        stability_details = consistency_results['subject_stability']['details']
        subject_pairs = [f"{d['subject_pair'][0][:8]}\nvs\n{d['subject_pair'][1][:8]}" 
                        for d in stability_details]
        consistency_rates = [d['consistency_rate'] for d in stability_details]
        comparison_counts = [d['total_comparisons'] for d in stability_details]
        
        # Create scatter plot
        scatter = ax.scatter(range(len(subject_pairs)), consistency_rates, 
                           s=[c*50 for c in comparison_counts], 
                           alpha=0.6, c=consistency_rates, cmap='RdYlGn')
        
        ax.set_xlabel('Subject Pair Comparisons')
        ax.set_ylabel('Consistency Rate')
        ax.set_title('Subject Preference Stability\n(Bubble size = number of comparisons)')
        ax.set_xticks(range(len(subject_pairs)))
        ax.set_xticklabels(subject_pairs, rotation=45, ha='right')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='70% threshold')
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Consistency Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'subject_stability_details.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'subject_stability_details.pdf'), bbox_inches='tight')
        plt.close()
    
    print(f"\nConsistency plots saved to: {plots_dir}")

def main():
    """Run consistency analysis on the focused representative experiment."""
    
    experiment_path = "/home/dcruz/model_preferences/experiments/20250719_232812_focused_representative"
    results_file = os.path.join(experiment_path, "results.csv")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    print("Analyzing preference consistency from focused representative experiment...")
    
    consistency_results = analyze_preference_consistency(results_file, experiment_path)
    
    print("\n" + "=" * 60)
    print("CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {experiment_path}")
    
    return consistency_results

if __name__ == "__main__":
    main()