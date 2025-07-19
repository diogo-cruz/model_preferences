#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_config

class BradleyTerryAnalysis:
    """Bradley-Terry model for pairwise comparison analysis."""
    
    def __init__(self):
        self.config = get_config()
        self.task_strengths = {}
        self.convergence_achieved = False
        self.log_likelihood = None
    
    def fit(self, comparison_data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit Bradley-Terry model to pairwise comparison data.
        
        Args:
            comparison_data: DataFrame with columns 'winner_id', 'task_a_id', 'task_b_id'
            
        Returns:
            Dictionary mapping task_id to strength parameter
        """
        # Get unique tasks
        all_tasks = set()
        for _, row in comparison_data.iterrows():
            if pd.notna(row['task_a_id']) and pd.notna(row['task_b_id']):
                all_tasks.add(int(row['task_a_id']))
                all_tasks.add(int(row['task_b_id']))
        
        task_list = sorted(list(all_tasks))
        n_tasks = len(task_list)
        task_to_idx = {task: i for i, task in enumerate(task_list)}
        
        print(f"Fitting Bradley-Terry model for {n_tasks} tasks...")
        
        # Initialize parameters (log-strengths, last one fixed at 0 for identifiability)
        initial_params = np.zeros(n_tasks - 1)
        
        def negative_log_likelihood(params):
            # Set last parameter to 0 for identifiability
            full_params = np.append(params, 0)
            log_strengths = full_params
            
            ll = 0
            for _, row in comparison_data.iterrows():
                if pd.isna(row['winner_id']) or pd.isna(row['task_a_id']) or pd.isna(row['task_b_id']):
                    continue
                
                task_a = int(row['task_a_id'])
                task_b = int(row['task_b_id'])
                winner = int(row['winner_id'])
                
                if task_a not in task_to_idx or task_b not in task_to_idx:
                    continue
                
                idx_a = task_to_idx[task_a]
                idx_b = task_to_idx[task_b]
                
                # Bradley-Terry probability
                diff = log_strengths[idx_a] - log_strengths[idx_b]
                prob_a_wins = 1 / (1 + np.exp(-diff))
                
                if winner == task_a:
                    ll += np.log(prob_a_wins + 1e-10)  # Add small constant for numerical stability
                elif winner == task_b:
                    ll += np.log(1 - prob_a_wins + 1e-10)
            
            return -ll
        
        # Optimize
        max_iter = self.config.get('analysis.bradley_terry.max_iterations', 1000)
        gtol = self.config.get('analysis.bradley_terry.convergence_tolerance', 1e-6)
        
        result = minimize(
            negative_log_likelihood,
            initial_params,
            method='BFGS',
            options={
                'maxiter': int(max_iter),
                'gtol': float(gtol)
            }
        )
        
        self.convergence_achieved = result.success
        self.log_likelihood = -result.fun
        
        # Extract strengths (convert from log scale)
        full_params = np.append(result.x, 0)
        log_strengths = full_params
        strengths = np.exp(log_strengths)
        
        # Normalize so they sum to number of tasks
        strengths = strengths * n_tasks / np.sum(strengths)
        
        self.task_strengths = {task_list[i]: strengths[i] for i in range(n_tasks)}
        
        print(f"Convergence: {'✅ Achieved' if self.convergence_achieved else '❌ Failed'}")
        print(f"Log-likelihood: {self.log_likelihood:.2f}")
        
        return self.task_strengths
    
    def get_preference_ranking(self, task_metadata: Dict[int, Dict]) -> List[Tuple[str, float, int]]:
        """
        Get preference ranking with subject information.
        
        Args:
            task_metadata: Dictionary mapping task_id to task information
            
        Returns:
            List of (subject, strength, task_id) tuples sorted by strength
        """
        ranking = []
        for task_id, strength in self.task_strengths.items():
            if task_id in task_metadata:
                subject = task_metadata[task_id]['subject']
                ranking.append((subject, strength, task_id))
        
        return sorted(ranking, key=lambda x: x[1], reverse=True)

class TransitivityAnalysis:
    """Analysis of transitivity violations in pairwise comparisons."""
    
    def __init__(self):
        self.violations = []
        self.total_triplets = 0
    
    def check_transitivity(self, comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for transitivity violations (A > B, B > C, C > A).
        
        Args:
            comparison_data: DataFrame with comparison results
            
        Returns:
            Dictionary with transitivity analysis results
        """
        # Build preference matrix
        preferences = {}
        
        for _, row in comparison_data.iterrows():
            if pd.isna(row['winner_id']) or pd.isna(row['task_a_id']) or pd.isna(row['task_b_id']):
                continue
            
            task_a = int(row['task_a_id'])
            task_b = int(row['task_b_id'])
            winner = int(row['winner_id'])
            
            if winner == task_a:
                preferences[(task_a, task_b)] = preferences.get((task_a, task_b), 0) + 1
            elif winner == task_b:
                preferences[(task_b, task_a)] = preferences.get((task_b, task_a), 0) + 1
        
        # Find all tasks
        all_tasks = set()
        for (a, b) in preferences.keys():
            all_tasks.add(a)
            all_tasks.add(b)
        
        task_list = sorted(list(all_tasks))
        
        # Check all triplets for violations
        violations = []
        total_triplets = 0
        
        for i, a in enumerate(task_list):
            for j, b in enumerate(task_list[i+1:], i+1):
                for k, c in enumerate(task_list[j+1:], j+1):
                    total_triplets += 1
                    
                    # Get preferences for this triplet
                    ab_pref = preferences.get((a, b), 0) - preferences.get((b, a), 0)
                    bc_pref = preferences.get((b, c), 0) - preferences.get((c, b), 0)
                    ca_pref = preferences.get((c, a), 0) - preferences.get((a, c), 0)
                    
                    # Check for violations (all three should not be positive)
                    if ab_pref > 0 and bc_pref > 0 and ca_pref > 0:
                        violations.append((a, b, c, 'A>B>C>A'))
                    elif ab_pref < 0 and bc_pref < 0 and ca_pref < 0:
                        violations.append((a, b, c, 'A<B<C<A'))
        
        self.violations = violations
        self.total_triplets = total_triplets
        
        violation_rate = len(violations) / max(total_triplets, 1)
        
        return {
            'total_triplets': total_triplets,
            'violations': len(violations),
            'violation_rate': violation_rate,
            'violation_details': violations
        }

class PositionBiasAnalysis:
    """Analysis of position bias in task presentation."""
    
    def analyze_position_bias(self, comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze whether there's bias toward tasks presented first (A) vs second (B).
        
        Args:
            comparison_data: DataFrame with comparison results
            
        Returns:
            Dictionary with position bias analysis
        """
        # Count choices by position
        valid_choices = comparison_data[comparison_data['choice'].notna()]
        
        choice_counts = valid_choices['choice'].value_counts()
        total_choices = len(valid_choices)
        
        if total_choices == 0:
            return {'error': 'No valid choices found'}
        
        choice_a = choice_counts.get('A', 0)
        choice_b = choice_counts.get('B', 0)
        
        # Statistical test for position bias (binomial test)
        # H0: P(choose A) = 0.5 (no position bias)
        try:
            # Try newer scipy version first
            p_value = stats.binomtest(choice_a, total_choices, 0.5).pvalue
        except AttributeError:
            # Fall back to older scipy version
            p_value = stats.binom_test(choice_a, total_choices, 0.5)
        
        return {
            'total_choices': total_choices,
            'choice_a': choice_a,
            'choice_b': choice_b,
            'choice_a_rate': choice_a / total_choices,
            'choice_b_rate': choice_b / total_choices,
            'bias_test_p_value': p_value,
            'significant_bias': p_value < 0.05
        }

def run_comprehensive_analysis(experiment_path: str) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis on experiment results.
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 60)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Load data
    results_file = f"{experiment_path}/results.csv"
    metadata_file = f"{experiment_path}/metadata.json"
    
    try:
        results_df = pd.read_csv(results_file)
        print(f"✅ Loaded {len(results_df)} comparison results")
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return {}
    
    # Basic statistics
    total_comparisons = len(results_df)
    successful_comparisons = len(results_df[results_df['success'] == True])
    
    print(f"\nBasic Statistics:")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Successful: {successful_comparisons} ({successful_comparisons/total_comparisons*100:.1f}%)")
    
    # Position bias analysis
    print(f"\n1. Position Bias Analysis:")
    bias_analyzer = PositionBiasAnalysis()
    bias_results = bias_analyzer.analyze_position_bias(results_df)
    
    if 'error' not in bias_results:
        print(f"  Choice A rate: {bias_results['choice_a_rate']:.3f}")
        print(f"  Choice B rate: {bias_results['choice_b_rate']:.3f}")
        print(f"  Bias test p-value: {bias_results['bias_test_p_value']:.6f}")
        print(f"  Significant bias: {'Yes' if bias_results['significant_bias'] else 'No'}")
    
    # Bradley-Terry analysis
    print(f"\n2. Bradley-Terry Model:")
    bt_analyzer = BradleyTerryAnalysis()
    
    # Create task metadata for ranking
    task_metadata = {}
    for _, row in results_df.iterrows():
        if pd.notna(row['task_a_id']):
            task_metadata[int(row['task_a_id'])] = {'subject': row['task_a_subject']}
        if pd.notna(row['task_b_id']):
            task_metadata[int(row['task_b_id'])] = {'subject': row['task_b_subject']}
    
    task_strengths = bt_analyzer.fit(results_df)
    ranking = bt_analyzer.get_preference_ranking(task_metadata)
    
    print(f"  Top 5 preferred subjects:")
    for i, (subject, strength, task_id) in enumerate(ranking[:5]):
        print(f"    {i+1}. {subject}: {strength:.3f}")
    
    # Transitivity analysis
    print(f"\n3. Transitivity Analysis:")
    trans_analyzer = TransitivityAnalysis()
    trans_results = trans_analyzer.check_transitivity(results_df)
    
    print(f"  Total triplets: {trans_results['total_triplets']}")
    print(f"  Violations: {trans_results['violations']}")
    print(f"  Violation rate: {trans_results['violation_rate']:.3f}")
    
    # Compile results
    analysis_results = {
        'basic_stats': {
            'total_comparisons': total_comparisons,
            'successful_comparisons': successful_comparisons,
            'success_rate': successful_comparisons / total_comparisons
        },
        'position_bias': bias_results,
        'bradley_terry': {
            'task_strengths': task_strengths,
            'ranking': ranking,
            'convergence': bt_analyzer.convergence_achieved,
            'log_likelihood': bt_analyzer.log_likelihood
        },
        'transitivity': trans_results
    }
    
    return analysis_results