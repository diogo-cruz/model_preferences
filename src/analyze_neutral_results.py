#!/usr/bin/env python3

"""
Analyze the neutral large-scale experiment results that completed successfully.
"""

import sys
import os
sys.path.append('/home/dcruz/model_preferences/src')

import pandas as pd
import numpy as np
from improved_plotting import create_comprehensive_analysis_plots

def analyze_neutral_results():
    """Analyze the completed neutral experiment results."""
    
    results_path = "/home/dcruz/model_preferences/experiments/20250720_030834_neutral_large_scale_40tasks_5x/results.csv"
    plots_dir = "/home/dcruz/model_preferences/experiments/20250720_030834_neutral_large_scale_40tasks_5x/plots"
    
    print("Loading neutral large-scale experiment results...")
    df = pd.read_csv(results_path)
    
    # Add the 'order' column for compatibility
    if 'order' not in df.columns and 'actual_order' in df.columns:
        df['order'] = df['actual_order']
    
    successful = df[df['success'] == True]
    
    print(f"Total comparisons: {len(df)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(df):.1%})")
    
    # Basic statistics
    if len(successful) > 0:
        choice_a_rate = (successful['choice'] == 'A').mean()
        print(f"Choice A rate: {choice_a_rate:.1%}")
        print(f"Position bias effect: {abs(choice_a_rate - 0.5)*2:.1%}")
        
        # Randomization check
        ab_orders = successful[successful['order'] == 'AB']
        ba_orders = successful[successful['order'] == 'BA']
        print(f"AB orders: {len(ab_orders)} ({len(ab_orders)/len(successful):.1%})")
        print(f"BA orders: {len(ba_orders)} ({len(ba_orders)/len(successful):.1%})")
        
        if len(ab_orders) > 0 and len(ba_orders) > 0:
            ab_choice_a = (ab_orders['choice'] == 'A').mean()
            ba_choice_a = (ba_orders['choice'] == 'A').mean()
            print(f"AB choice A rate: {ab_choice_a:.1%}")
            print(f"BA choice A rate: {ba_choice_a:.1%}")
            print(f"Position bias difference: {abs(ab_choice_a - ba_choice_a):.1%}")
    
    # Create plots
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nGenerating plots...")
    
    analysis_results = create_comprehensive_analysis_plots(
        successful, plots_dir, "Neutral 5x Scale 40 Tasks"
    )
    
    print(f"Plots saved to: {plots_dir}")
    
    # Print variance analysis
    if analysis_results and 'variance_analysis' in analysis_results:
        va = analysis_results['variance_analysis']
        print(f"\nVariance Analysis Results:")
        print(f"  Mean error bar: {va['mean_error_bar']:.3f}")
        print(f"  Previous experiments:")
        print(f"    - Initial (20 tasks): 0.238")
        print(f"    - Large sample (30 tasks): 0.167")
        print(f"    - Neutral 5x (40 tasks): {va['mean_error_bar']:.3f}")
        
        improvement_from_initial = ((0.238 - va['mean_error_bar']) / 0.238 * 100)
        improvement_from_large = ((0.167 - va['mean_error_bar']) / 0.167 * 100)
        print(f"  Improvement from initial: {improvement_from_initial:.1f}% reduction")
        print(f"  Improvement from large sample: {improvement_from_large:.1f}% reduction")
        
        print(f"  Overall recommendation: {va['overall_recommendation']}")
        
        if va['individual_recommendations']:
            print(f"  Specific recommendations:")
            for rec in va['individual_recommendations']:
                print(f"    - {rec}")
        else:
            print(f"  ✅ Sample sizes appear adequate!")
    
    return analysis_results

if __name__ == "__main__":
    analyze_neutral_results()