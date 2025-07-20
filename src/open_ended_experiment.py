#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/dcruz/model_preferences/src')

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any
import numpy as np

from task_selection import TaskSelector
from open_ended_client import OpenEndedModelClient
from config import get_config

class OpenEndedExperiment:
    """
    Experiment class for running open-ended task preference comparisons.
    """
    
    def __init__(self, n_tasks: int = 20, seed: int = 42):
        """
        Initialize the open-ended experiment.
        
        Args:
            n_tasks: Number of tasks to sample for comparison
            seed: Random seed for reproducibility
        """
        self.config = get_config()
        self.n_tasks = n_tasks
        self.seed = seed
        self.task_selector = TaskSelector(seed=seed)
        self.client = OpenEndedModelClient()
        self.tasks = []
        self.results = []
        self.experiment_name = f"open_ended_comparison_{n_tasks}tasks"
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"/home/dcruz/model_preferences/experiments/{timestamp}_{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        
        print(f"Experiment directory: {self.experiment_dir}")
    
    def generate_representative_pairs(self, max_comparisons: int = 150) -> List[tuple]:
        """
        Generate representative pairs for comparison (same as focused experiment).
        
        Args:
            max_comparisons: Maximum number of comparisons to generate
            
        Returns:
            List of (task_i, task_j, order) tuples
        """
        pairs = []
        n_tasks = len(self.tasks)
        
        # Generate pairs with limited scope to stay within max_comparisons
        for i in range(n_tasks):
            # Compare each task with next few tasks (not all combinations)
            for j in range(i + 1, min(i + 4, n_tasks)):  # Limit to next 3 tasks
                pairs.append((i, j, 'AB'))
                pairs.append((j, i, 'BA'))  # Include reverse for bias control
        
        # If we have too many pairs, truncate
        if len(pairs) > max_comparisons:
            pairs = pairs[:max_comparisons]
        
        print(f"Generated {len(pairs)} representative comparison pairs")
        return pairs
    
    def run_experiment(self, max_comparisons: int = 150) -> List[Dict[str, Any]]:
        """
        Run the complete open-ended experiment.
        
        Args:
            max_comparisons: Maximum number of comparisons to run
            
        Returns:
            List of experiment results
        """
        print("=" * 60)
        print("OPEN-ENDED TASK PREFERENCE EXPERIMENT")
        print("=" * 60)
        print(f"Model: {self.client.model_name}")
        print(f"Tasks: {self.n_tasks}")
        print(f"Max comparisons: {max_comparisons}")
        print(f"Seed: {self.seed}")
        
        # Sample tasks
        print("\n1. Sampling tasks...")
        self.tasks = self.task_selector.sample_tasks(self.n_tasks)
        print(f"Selected {len(self.tasks)} tasks from {len(set(t['subject'] for t in self.tasks))} subjects")
        
        # Generate pairs
        print("\n2. Generating comparison pairs...")
        pairs = self.generate_representative_pairs(max_comparisons)
        
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
                'response_time': result['response_time'],
                'api_call_id': result['api_call_id'],
                'timestamp': datetime.now().isoformat(),
                'winner_id': winner_idx,
                'winner_subject': winner_subject
            }
            
            self.results.append(comparison_result)
            
            # Brief progress update
            if (i + 1) % 25 == 0:
                success_rate = sum(1 for r in self.results if r['success']) / len(self.results)
                print(f"  Progress: {i+1}/{len(pairs)} ({success_rate:.1%} success rate)")
        
        # Save results
        self.save_results()
        
        # Generate analysis and plots
        self.analyze_and_plot()
        
        print(f"\n✅ Experiment completed!")
        print(f"Results saved to: {self.experiment_dir}")
        
        return self.results
    
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
            'experiment_type': 'open_ended_comparison'
        }
        
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved: {results_path}")
        print(f"Metadata saved: {metadata_path}")
    
    def analyze_and_plot(self):
        """Analyze results and generate plots."""
        
        if not self.results:
            print("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        successful = df[df['success'] == True]
        
        if successful.empty:
            print("No successful comparisons to analyze")
            return
        
        plots_dir = os.path.join(self.experiment_dir, "plots")
        
        print(f"\n5. Generating analysis plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Position Bias Analysis
        self._plot_position_bias(successful, plots_dir)
        
        # 2. Subject Preferences
        self._plot_subject_preferences(successful, plots_dir)
        
        # 3. Response Times
        self._plot_response_times(successful, plots_dir)
        
        # 4. Success Rates
        self._plot_success_rates(df, plots_dir)
        
        print(f"Plots saved to: {plots_dir}")
    
    def _plot_position_bias(self, df, plots_dir):
        """Plot position bias analysis."""
        
        choice_counts = df['choice'].value_counts()
        choice_a_rate = choice_counts.get('A', 0) / len(df)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall choice distribution
        ax1.pie([choice_counts.get('A', 0), choice_counts.get('B', 0)], 
                labels=['Choice A', 'Choice B'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Overall Choice Distribution\n(Open-Ended Questions)')
        
        # Choice by order
        ab_data = df[df['order'] == 'AB']
        ba_data = df[df['order'] == 'BA']
        
        ab_choice_a = (ab_data['choice'] == 'A').mean() if len(ab_data) > 0 else 0
        ba_choice_a = (ba_data['choice'] == 'A').mean() if len(ba_data) > 0 else 0
        
        orders = ['AB Order', 'BA Order']
        choice_a_rates = [ab_choice_a, ba_choice_a]
        
        bars = ax2.bar(orders, choice_a_rates, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax2.set_ylabel('Choice A Rate')
        ax2.set_title('Position Bias Analysis\n(Open-Ended Questions)')
        ax2.set_ylim(0, 1)
        
        for bar, rate in zip(bars, choice_a_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'position_bias_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'position_bias_analysis.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_subject_preferences(self, df, plots_dir):
        """Plot subject preference analysis."""
        
        winners = df[df['winner_subject'].notna()]
        subject_wins = Counter(winners['winner_subject'])
        
        if not subject_wins:
            return
        
        # Get top 10 subjects
        top_subjects = subject_wins.most_common(10)
        subjects, wins = zip(*top_subjects)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(subjects)), wins, color='lightgreen', alpha=0.7)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels([s.replace('_', ' ').title() for s in subjects])
        ax.set_xlabel('Number of Wins')
        ax.set_title('Subject Preferences\n(Open-Ended Questions - Top 10)')
        
        # Add value labels
        for i, (bar, win_count) in enumerate(zip(bars, wins)):
            width = bar.get_width()
            percentage = (win_count / len(winners)) * 100
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{win_count} ({percentage:.1f}%)', 
                   ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'subject_preferences.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'subject_preferences.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_response_times(self, df, plots_dir):
        """Plot response time analysis."""
        
        response_times = df['response_time'].dropna()
        
        if response_times.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(response_times, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(response_times.mean(), color='red', linestyle='--', 
                  label=f'Mean: {response_times.mean():.3f}s')
        ax.axvline(response_times.median(), color='orange', linestyle='--', 
                  label=f'Median: {response_times.median():.3f}s')
        ax.set_xlabel('Response Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Time Distribution\n(Open-Ended Questions)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_times.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'response_times.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_success_rates(self, df, plots_dir):
        """Plot API success rate analysis."""
        
        success_rate = df['success'].mean()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.pie([success_rate, 1 - success_rate], 
               labels=['Successful', 'Failed'], 
               autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'],
               startangle=90)
        ax.set_title(f'API Success Rate\n(Open-Ended Questions)\nTotal: {len(df)} calls')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'success_rates.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'success_rates.pdf'), bbox_inches='tight')
        plt.close()

def main():
    """Run open-ended experiment with default parameters."""
    
    # Use same parameters as focused experiment for comparison
    experiment = OpenEndedExperiment(n_tasks=20, seed=42)
    results = experiment.run_experiment(max_comparisons=150)
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    print(f"\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total comparisons: {len(results)}")
    print(f"Successful: {len(successful_results)} ({len(successful_results)/len(results):.1%})")
    print(f"Failed: {len(results) - len(successful_results)}")
    
    if successful_results:
        choices = [r['choice'] for r in successful_results if r['choice']]
        if choices:
            choice_a_rate = choices.count('A') / len(choices)
            print(f"Choice A rate: {choice_a_rate:.1%}")
            print(f"Choice B rate: {1-choice_a_rate:.1%}")
    
    return results

if __name__ == "__main__":
    main()