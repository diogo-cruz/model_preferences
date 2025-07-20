#!/usr/bin/env python3

"""
Large-scale neutral experiment with 5x more samples and unbiased prompting.
Implements randomized task order and minimal prompting per INSTRUCTIONS.md.
"""

import sys
import os
sys.path.append('/home/dcruz/model_preferences/src')

import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter

from task_selection import TaskSelector
from neutral_open_ended_client import NeutralOpenEndedModelClient
from config import get_config
from improved_plotting import create_comprehensive_analysis_plots

class NeutralLargeScaleExperiment:
    """
    Large-scale neutral experiment with 5x samples and unbiased prompting.
    """
    
    def __init__(self, n_tasks: int = 40, seed: int = 42):
        """
        Initialize the neutral large-scale experiment.
        
        Args:
            n_tasks: Number of tasks to sample (increased from 30 to 40)
            seed: Random seed for reproducibility
        """
        self.config = get_config()
        self.n_tasks = n_tasks
        self.seed = seed
        self.task_selector = TaskSelector(seed=seed)
        self.client = NeutralOpenEndedModelClient()
        self.tasks = []
        self.results = []
        self.experiment_name = f"neutral_large_scale_{n_tasks}tasks_5x"
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"/home/dcruz/model_preferences/experiments/{timestamp}_{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        
        print(f"Neutral Large-Scale Experiment directory: {self.experiment_dir}")
    
    def generate_comprehensive_pairs(self, target_comparisons: int = 2000) -> List[tuple]:
        """
        Generate comprehensive pairs for maximum statistical power (5x increase).
        
        Args:
            target_comparisons: Target number of comparisons (5x previous ~400)
            
        Returns:
            List of (task_i, task_j, order) tuples
        """
        pairs = []
        n_tasks = len(self.tasks)
        
        # Calculate how many rounds of comparisons we need
        max_unique_pairs = n_tasks * (n_tasks - 1) // 2  # All possible unique pairs
        comparisons_per_pair = max(1, target_comparisons // (max_unique_pairs * 2))  # *2 for AB/BA
        
        print(f"Generating {comparisons_per_pair} rounds of all {max_unique_pairs} unique pairs")
        
        # Generate multiple rounds of comprehensive comparisons
        for round_num in range(comparisons_per_pair):
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    # Add both orderings for bias control
                    pairs.append((i, j, 'AB'))
                    pairs.append((j, i, 'BA'))
        
        # If we still need more pairs to reach target, add additional rounds of popular pairs
        while len(pairs) < target_comparisons:
            # Add another round of all pairs
            for i in range(n_tasks):
                for j in range(i + 1, min(i + 6, n_tasks)):  # Limit to avoid explosion
                    if len(pairs) >= target_comparisons:
                        break
                    pairs.append((i, j, 'AB'))
                    if len(pairs) >= target_comparisons:
                        break
                    pairs.append((j, i, 'BA'))
                if len(pairs) >= target_comparisons:
                    break
        
        # Trim to exact target
        pairs = pairs[:target_comparisons]
        
        print(f"Generated {len(pairs)} comprehensive comparison pairs (5x scale)")
        return pairs
    
    def run_experiment(self, target_comparisons: int = 2000) -> List[Dict[str, Any]]:
        """
        Run the neutral large-scale experiment with 5x samples.
        
        Args:
            target_comparisons: Target number of comparisons (5x increase)
            
        Returns:
            List of experiment results
        """
        print("=" * 80)
        print("NEUTRAL LARGE-SCALE EXPERIMENT (5X SAMPLES)")
        print("=" * 80)
        print(f"Model: {self.client.model_name}")
        print(f"Tasks: {self.n_tasks}")
        print(f"Target comparisons: {target_comparisons}")
        print(f"Seed: {self.seed}")
        print(f"Key improvements:")
        print(f"  - Neutral prompting: 'Pick which one you want to do'")
        print(f"  - Randomized task order for bias control")
        print(f"  - 5x sample size for error bar reduction")
        
        # Sample more tasks
        print("\n1. Sampling tasks...")
        self.tasks = self.task_selector.sample_tasks(self.n_tasks)
        print(f"Selected {len(self.tasks)} tasks from {len(set(t['subject'] for t in self.tasks))} subjects")
        
        # Generate comprehensive pairs
        print("\n2. Generating comprehensive comparison pairs...")
        pairs = self.generate_comprehensive_pairs(target_comparisons)
        
        # Test API connection
        print("\n3. Testing API connection...")
        if not self.client.test_api_connection():
            raise Exception("API connection failed")
        print("API connection successful")
        
        # Run comparisons
        print(f"\n4. Running {len(pairs)} neutral comparisons with randomization...")
        self.results = []
        
        for i, (task_a_idx, task_b_idx, intended_order) in enumerate(pairs):
            task_a = self.tasks[task_a_idx]
            task_b = self.tasks[task_b_idx]
            
            if (i + 1) % 100 == 0:  # Progress every 100
                print(f"Progress {i+1}/{len(pairs)}: Task {task_a_idx} vs {task_b_idx}")
            
            # Get model choice using neutral prompting with randomization
            result = self.client.get_task_choice(task_a, task_b)
            
            # Determine winner based on final mapped choice
            if result['success'] and result['choice']:
                if result['choice'] == 'A':
                    winner_idx = task_a_idx
                    winner_subject = task_a['subject']
                else:
                    winner_idx = task_b_idx
                    winner_subject = task_b['subject']
            else:
                winner_idx = None
                winner_subject = None
            
            # Store result with randomization metadata
            comparison_result = {
                'task_a_id': task_a_idx,
                'task_b_id': task_b_idx,
                'task_a_subject': task_a['subject'],
                'task_b_subject': task_b['subject'],
                'intended_order': intended_order,  # What we intended
                'actual_order': result['actual_order'],  # What was actually presented
                'order': result['actual_order'],  # For compatibility with plotting
                'choice': result['choice'],  # Final mapped choice
                'raw_choice': result['raw_choice'],  # Raw model response
                'raw_response': result['raw_response'],
                'success': result['success'],
                'error': result['error'],
                'response_time': result['response_time'],
                'api_call_id': result['api_call_id'],
                'timestamp': datetime.now().isoformat(),
                'winner_id': winner_idx,
                'winner_subject': winner_subject,
                'prompt_type': 'neutral_randomized'
            }
            
            self.results.append(comparison_result)
            
            # Progress updates
            if (i + 1) % 500 == 0:
                success_rate = sum(1 for r in self.results if r['success']) / len(self.results)
                choice_a_rate = sum(1 for r in self.results if r['success'] and r['choice'] == 'A') / sum(1 for r in self.results if r['success'])
                print(f"  Progress: {i+1}/{len(pairs)} ({success_rate:.1%} success, {choice_a_rate:.1%} choice A)")
        
        # Save results
        self.save_results()
        
        # Generate analysis and plots with variance analysis
        analysis_results = self.analyze_and_plot()
        
        print(f"\n✅ Neutral Large-Scale Experiment completed!")
        print(f"Results saved to: {self.experiment_dir}")
        
        return self.results, analysis_results
    
    def save_results(self):
        """Save experiment results and metadata."""
        
        # Save results CSV
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.experiment_dir, "results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Calculate summary statistics
        successful_results = [r for r in self.results if r['success']]
        subject_distribution = Counter(task['subject'] for task in self.tasks)
        
        # Calculate randomization statistics
        randomization_stats = {}
        if successful_results:
            ab_orders = [r for r in successful_results if r['actual_order'] == 'AB']
            ba_orders = [r for r in successful_results if r['actual_order'] == 'BA']
            randomization_stats = {
                'ab_count': len(ab_orders),
                'ba_count': len(ba_orders),
                'randomization_balance': len(ab_orders) / (len(ab_orders) + len(ba_orders)) if (ab_orders or ba_orders) else 0
            }
        
        # Save metadata
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'n_tasks': self.n_tasks,
            'seed': self.seed,
            'model_name': self.client.model_name,
            'experiment_path': self.experiment_dir,
            'total_comparisons': len(self.results),
            'successful_comparisons': len(successful_results),
            'failed_comparisons': len(self.results) - len(successful_results),
            'total_tasks': len(self.tasks),
            'unique_subjects': len(subject_distribution),
            'subject_distribution': dict(subject_distribution),
            'seed_used': self.seed,
            'experiment_type': 'neutral_large_scale_5x',
            'improvements': {
                'neutral_prompting': 'Removed biasing factors from prompt',
                'task_randomization': 'Randomized task order to control position bias',
                'sample_size_5x': 'Increased sample size 5x to reduce error bars',
                'simplified_prompt': 'Just says "Pick which one you want to do"'
            },
            'randomization_stats': randomization_stats,
            'purpose': '5x sample size with neutral prompting per INSTRUCTIONS.md'
        }
        
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved: {results_path}")
        print(f"Metadata saved: {metadata_path}")
    
    def analyze_and_plot(self):
        """Analyze results and generate plots with variance analysis."""
        
        if not self.results:
            print("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        successful = df[df['success'] == True]
        
        if successful.empty:
            print("No successful comparisons to analyze")
            return
        
        plots_dir = os.path.join(self.experiment_dir, "plots")
        
        print(f"\n5. Generating comprehensive analysis with variance assessment...")
        
        # Use improved plotting module with error bars and ratios
        analysis_results = create_comprehensive_analysis_plots(
            successful, plots_dir, f"Neutral 5x Scale {self.n_tasks} Tasks"
        )
        
        print(f"Plots saved to: {plots_dir}")
        
        # Print randomization analysis
        ab_orders = successful[successful['actual_order'] == 'AB']
        ba_orders = successful[successful['actual_order'] == 'BA']
        ab_choice_a_rate = (ab_orders['choice'] == 'A').mean() if len(ab_orders) > 0 else 0
        ba_choice_a_rate = (ba_orders['choice'] == 'A').mean() if len(ba_orders) > 0 else 0
        
        print(f"\nRandomization Analysis:")
        print(f"  AB orders: {len(ab_orders)} ({len(ab_orders)/len(successful):.1%})")
        print(f"  BA orders: {len(ba_orders)} ({len(ba_orders)/len(successful):.1%})")
        print(f"  AB choice A rate: {ab_choice_a_rate:.1%}")
        print(f"  BA choice A rate: {ba_choice_a_rate:.1%}")
        print(f"  Position bias difference: {abs(ab_choice_a_rate - ba_choice_a_rate):.1%}")
        
        # Print variance analysis recommendations
        if 'variance_analysis' in analysis_results:
            va = analysis_results['variance_analysis']
            print(f"\nVariance Analysis Results:")
            print(f"  Mean error bar: {va['mean_error_bar']:.3f}")
            print(f"  Variance to signal ratio: {va['variance_to_signal_ratio']:.3f}")
            print(f"  Sample size range: {va['min_sample_size']} to {va['max_sample_size']}")
            print(f"  Overall recommendation: {va['overall_recommendation']}")
            
            if va['individual_recommendations']:
                print(f"  Specific recommendations:")
                for rec in va['individual_recommendations']:
                    print(f"    - {rec}")
            else:
                print(f"  ✅ Sample sizes appear adequate!")
        
        return analysis_results

def main():
    """Run neutral large-scale experiment with 5x samples."""
    
    # Significant increase in sample size (5x from ~400 to 2000)
    experiment = NeutralLargeScaleExperiment(n_tasks=40, seed=42)
    results, analysis = experiment.run_experiment(target_comparisons=2000)
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    print(f"\n" + "=" * 80)
    print("NEUTRAL LARGE-SCALE EXPERIMENT SUMMARY (5X)")
    print("=" * 80)
    print(f"Total comparisons: {len(results)}")
    print(f"Successful: {len(successful_results)} ({len(successful_results)/len(results):.1%})")
    print(f"Failed: {len(results) - len(successful_results)}")
    print(f"Tasks analyzed: {experiment.n_tasks}")
    print(f"Sample size increase: 5x (from ~400 to {len(results)})")
    
    if successful_results:
        choices = [r['choice'] for r in successful_results if r['choice']]
        if choices:
            choice_a_rate = choices.count('A') / len(choices)
            print(f"Choice A rate: {choice_a_rate:.1%}")
            print(f"Choice B rate: {1-choice_a_rate:.1%}")
            print(f"Position bias effect: {abs(choice_a_rate - 0.5)*2:.1%}")
    
    # Compare error bar improvement
    if analysis and 'variance_analysis' in analysis:
        va = analysis['variance_analysis']
        print(f"\nStatistical Power Assessment:")
        print(f"  Mean error bar: {va['mean_error_bar']:.3f}")
        print(f"  Previous (30 tasks, 384 comparisons): 0.167")
        print(f"  Improvement: {((0.167 - va['mean_error_bar']) / 0.167 * 100):.1f}% reduction" if va['mean_error_bar'] < 0.167 else f"  Change: {((va['mean_error_bar'] - 0.167) / 0.167 * 100):.1f}% increase")
        
        if va['overall_recommendation'] == 'samples_adequate':
            print(f"\n✅ SUCCESS: Achieved adequate sample sizes with neutral prompting!")
        elif va['overall_recommendation'] == 'consider_more_samples':
            print(f"\n✅ GOOD: Sample sizes much improved with neutral approach")
        else:
            print(f"\n⚠️  PARTIAL: Some improvement but may need even more samples")
    
    return results, analysis

if __name__ == "__main__":
    main()