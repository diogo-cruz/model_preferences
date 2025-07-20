#!/usr/bin/env python3

"""
Large sample experiment to address variance analysis recommendations.
Runs more comparisons to reduce error bars and improve statistical power.
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
from open_ended_client import OpenEndedModelClient
from config import get_config
from improved_plotting import create_comprehensive_analysis_plots

class LargeSampleExperiment:
    """
    Large sample experiment for improved statistical power.
    """
    
    def __init__(self, n_tasks: int = 30, seed: int = 42):
        """
        Initialize the large sample experiment.
        
        Args:
            n_tasks: Number of tasks to sample (increased from 20)
            seed: Random seed for reproducibility
        """
        self.config = get_config()
        self.n_tasks = n_tasks
        self.seed = seed
        self.task_selector = TaskSelector(seed=seed)
        self.client = OpenEndedModelClient()
        self.tasks = []
        self.results = []
        self.experiment_name = f"large_sample_open_ended_{n_tasks}tasks"
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"/home/dcruz/model_preferences/experiments/{timestamp}_{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        
        print(f"Large Sample Experiment directory: {self.experiment_dir}")
    
    def generate_expanded_pairs(self, max_comparisons: int = 400) -> List[tuple]:
        """
        Generate more comprehensive pairs for better statistical power.
        
        Args:
            max_comparisons: Maximum number of comparisons to generate (increased)
            
        Returns:
            List of (task_i, task_j, order) tuples
        """
        pairs = []
        n_tasks = len(self.tasks)
        
        # Generate more comprehensive pairwise comparisons
        for i in range(n_tasks):
            # Compare each task with more neighbors for better coverage
            for j in range(i + 1, min(i + 6, n_tasks)):  # Increased from 4 to 6
                pairs.append((i, j, 'AB'))
                pairs.append((j, i, 'BA'))  # Include reverse for bias control
                
                # Add additional cross-comparisons for popular subjects
                # This helps reduce error bars for frequently appearing subjects
                if j < i + 3:  # Inner comparisons get duplicated
                    pairs.append((i, j, 'AB'))
                    pairs.append((j, i, 'BA'))
        
        # If we have too many pairs, truncate but ensure balanced AB/BA
        if len(pairs) > max_comparisons:
            # Keep pairs balanced by ensuring even number
            target = (max_comparisons // 2) * 2
            pairs = pairs[:target]
        
        print(f"Generated {len(pairs)} expanded comparison pairs for better statistics")
        return pairs
    
    def run_experiment(self, max_comparisons: int = 400) -> List[Dict[str, Any]]:
        """
        Run the large sample experiment with increased comparisons.
        
        Args:
            max_comparisons: Maximum number of comparisons to run (increased)
            
        Returns:
            List of experiment results
        """
        print("=" * 80)
        print("LARGE SAMPLE OPEN-ENDED TASK PREFERENCE EXPERIMENT")
        print("=" * 80)
        print(f"Model: {self.client.model_name}")
        print(f"Tasks: {self.n_tasks}")
        print(f"Max comparisons: {max_comparisons}")
        print(f"Seed: {self.seed}")
        print(f"Objective: Reduce error bars and improve statistical power")
        
        # Sample more tasks
        print("\n1. Sampling more tasks...")
        self.tasks = self.task_selector.sample_tasks(self.n_tasks)
        print(f"Selected {len(self.tasks)} tasks from {len(set(t['subject'] for t in self.tasks))} subjects")
        
        # Generate expanded pairs
        print("\n2. Generating expanded comparison pairs...")
        pairs = self.generate_expanded_pairs(max_comparisons)
        
        # Test API connection
        print("\n3. Testing API connection...")
        if not self.client.test_api_connection():
            raise Exception("API connection failed")
        print("API connection successful")
        
        # Run comparisons
        print(f"\n4. Running {len(pairs)} comparisons...")
        self.results = []
        
        for i, (task_a_idx, task_b_idx, order) in enumerate(pairs):
            task_a = self.tasks[task_a_idx]
            task_b = self.tasks[task_b_idx]
            
            if (i + 1) % 50 == 0:  # Progress every 50 instead of 25
                print(f"Comparison {i+1}/{len(pairs)}: Task {task_a_idx} vs {task_b_idx} ({order})")
            
            # Get model choice
            result = self.client.get_task_choice(task_a, task_b)
            
            # Determine winner
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
            
            # Store result
            comparison_result = {
                'task_a_id': task_a_idx,
                'task_b_id': task_b_idx,
                'task_a_subject': task_a['subject'],
                'task_b_subject': task_b['subject'],
                'order': order,
                'choice': result['choice'],
                'raw_response': result['raw_response'],
                'success': result['success'],
                'error': result['error'],
                'response_time': result['response_time'],  # Keep for metadata, don't analyze
                'api_call_id': result['api_call_id'],
                'timestamp': datetime.now().isoformat(),
                'winner_id': winner_idx,
                'winner_subject': winner_subject
            }
            
            self.results.append(comparison_result)
            
            # Progress updates
            if (i + 1) % 100 == 0:
                success_rate = sum(1 for r in self.results if r['success']) / len(self.results)
                print(f"  Progress: {i+1}/{len(pairs)} ({success_rate:.1%} success rate)")
        
        # Save results
        self.save_results()
        
        # Generate analysis and plots with variance analysis
        analysis_results = self.analyze_and_plot()
        
        print(f"\n✅ Large Sample Experiment completed!")
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
            'experiment_type': 'large_sample_open_ended',
            'purpose': 'Reduce error bars and improve statistical power based on variance analysis'
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
            successful, plots_dir, f"Large Sample {self.n_tasks} Tasks"
        )
        
        print(f"Plots saved to: {plots_dir}")
        
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
                print(f"  ✅ Sample sizes appear adequate for current analysis!")
        
        return analysis_results

def main():
    """Run large sample experiment with increased statistical power."""
    
    # Increased sample size based on variance analysis recommendations
    experiment = LargeSampleExperiment(n_tasks=30, seed=42)
    results, analysis = experiment.run_experiment(max_comparisons=400)
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    print(f"\n" + "=" * 80)
    print("LARGE SAMPLE EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total comparisons: {len(results)}")
    print(f"Successful: {len(successful_results)} ({len(successful_results)/len(results):.1%})")
    print(f"Failed: {len(results) - len(successful_results)}")
    print(f"Tasks analyzed: {experiment.n_tasks}")
    
    if successful_results:
        choices = [r['choice'] for r in successful_results if r['choice']]
        if choices:
            choice_a_rate = choices.count('A') / len(choices)
            print(f"Choice A rate: {choice_a_rate:.1%}")
            print(f"Choice B rate: {1-choice_a_rate:.1%}")
            print(f"Position bias effect: {abs(choice_a_rate - 0.5)*2:.1%}")
    
    # Compare with variance analysis recommendations
    if analysis and 'variance_analysis' in analysis:
        va = analysis['variance_analysis']
        if va['overall_recommendation'] == 'samples_adequate':
            print(f"\n✅ SUCCESS: Achieved adequate sample sizes!")
        elif va['overall_recommendation'] == 'consider_more_samples':
            print(f"\n⚠️  MARGINAL: Sample sizes improved but could be better")
        else:
            print(f"\n❌ INSUFFICIENT: Still need more samples for robust statistics")
    
    return results, analysis

if __name__ == "__main__":
    main()