#!/usr/bin/env python3

import os
import json
import pandas as pd
from datetime import datetime
from itertools import combinations
from typing import List, Dict, Any, Tuple
from task_selection import TaskSelector
from api_client import ModelClient
from config import get_config

class PairwiseExperiment:
    def __init__(self, n_tasks: int = None, seed: int = None, model_name: str = None):
        """
        Initialize the pairwise comparison experiment.
        
        Args:
            n_tasks: Number of tasks to sample from MMLU (from config if None)
            seed: Random seed for reproducibility (from config if None)
            model_name: Name of the model to use (from config if None)
        """
        config = get_config()
        self.n_tasks = n_tasks or config.task_count
        self.seed = seed or config.experiment_seed
        self.model_name = model_name or config.model_name
        
        self.task_selector = TaskSelector(seed=self.seed)
        self.model_client = ModelClient(model_name=self.model_name)
        
        self.tasks = []
        self.results = []
        self.experiment_metadata = {}
        
    def setup_experiment(self, experiment_name: str = None) -> str:
        """
        Set up experiment folder and metadata.
        
        Args:
            experiment_name: Optional name for the experiment
            
        Returns:
            Path to the experiment folder
        """
        # Create experiment folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            folder_name = f"{timestamp}_{experiment_name}"
        else:
            folder_name = f"{timestamp}_pairwise_comparison"
        
        config = get_config()
        experiment_path = os.path.join(config.get_project_path('experiments'), folder_name)
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(f"{experiment_path}/plots", exist_ok=True)
        
        # Initialize metadata
        self.experiment_metadata = {
            'experiment_name': experiment_name or 'pairwise_comparison',
            'timestamp': timestamp,
            'n_tasks': self.n_tasks,
            'seed': self.seed,
            'model_name': self.model_name,
            'experiment_path': experiment_path,
            'total_comparisons': 0,
            'successful_comparisons': 0,
            'failed_comparisons': 0
        }
        
        return experiment_path
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load and sample tasks from MMLU dataset."""
        print(f"Sampling {self.n_tasks} tasks with seed {self.seed}")
        self.tasks = self.task_selector.sample_tasks(self.n_tasks)
        
        # Update metadata with task information
        task_summary = self.task_selector.get_task_summary()
        self.experiment_metadata.update(task_summary)
        
        return self.tasks
    
    def generate_pairs(self, include_reverse: bool = True) -> List[Tuple[int, int, str]]:
        """
        Generate all pairwise combinations of tasks.
        
        Args:
            include_reverse: If True, include both (A,B) and (B,A) orderings
            
        Returns:
            List of tuples (task_a_id, task_b_id, order) where order is 'AB' or 'BA'
        """
        pairs = []
        
        # Generate all unique pairs
        for i, j in combinations(range(len(self.tasks)), 2):
            pairs.append((i, j, 'AB'))
            if include_reverse:
                pairs.append((j, i, 'BA'))
        
        print(f"Generated {len(pairs)} pairwise comparisons")
        self.experiment_metadata['total_comparisons'] = len(pairs)
        
        return pairs
    
    def run_comparison(self, task_a_id: int, task_b_id: int, order: str) -> Dict[str, Any]:
        """
        Run a single pairwise comparison.
        
        Args:
            task_a_id: ID of first task
            task_b_id: ID of second task
            order: 'AB' or 'BA' indicating presentation order
            
        Returns:
            Dictionary with comparison results
        """
        task_a = self.tasks[task_a_id]
        task_b = self.tasks[task_b_id]
        
        # Format tasks for prompting
        task_a_text = self.task_selector.format_task_for_prompt(task_a)
        task_b_text = self.task_selector.format_task_for_prompt(task_b)
        
        # Get model's choice
        result = self.model_client.get_task_choice(task_a_text, task_b_text)
        
        # Compile results
        comparison_result = {
            'task_a_id': task_a_id,
            'task_b_id': task_b_id,
            'task_a_subject': task_a['subject'],
            'task_b_subject': task_b['subject'],
            'order': order,
            'choice': result['choice'],
            'raw_response': result['raw_response'],
            'success': result['success'],
            'error': result['error'],
            'response_time': result['response_time'],
            'api_call_id': result['api_call_id'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine winner based on choice and order
        if result['success'] and result['choice']:
            if result['choice'] == 'A':
                comparison_result['winner_id'] = task_a_id
                comparison_result['winner_subject'] = task_a['subject']
            elif result['choice'] == 'B':
                comparison_result['winner_id'] = task_b_id
                comparison_result['winner_subject'] = task_b['subject']
            else:
                comparison_result['winner_id'] = None
                comparison_result['winner_subject'] = None
        else:
            comparison_result['winner_id'] = None
            comparison_result['winner_subject'] = None
        
        return comparison_result
    
    def run_experiment(self, max_comparisons: int = None, experiment_name: str = None) -> str:
        """
        Run the full pairwise comparison experiment.
        
        Args:
            max_comparisons: Maximum number of comparisons to run (for testing)
            experiment_name: Optional name for the experiment
            
        Returns:
            Path to the experiment folder
        """
        # Setup
        experiment_path = self.setup_experiment(experiment_name)
        print(f"Experiment folder: {experiment_path}")
        
        # Load tasks
        self.load_tasks()
        
        # Generate pairs
        pairs = self.generate_pairs(include_reverse=True)
        
        # Limit pairs for testing if specified
        if max_comparisons:
            pairs = pairs[:max_comparisons]
            print(f"Limited to {max_comparisons} comparisons for testing")
        
        # Run comparisons
        print(f"Running {len(pairs)} pairwise comparisons...")
        results = []
        
        for i, (task_a_id, task_b_id, order) in enumerate(pairs):
            print(f"Comparison {i+1}/{len(pairs)}: Task {task_a_id} vs Task {task_b_id} ({order})")
            
            result = self.run_comparison(task_a_id, task_b_id, order)
            results.append(result)
            
            # Track success/failure
            if result['success']:
                self.experiment_metadata['successful_comparisons'] += 1
            else:
                self.experiment_metadata['failed_comparisons'] += 1
                print(f"  Failed: {result['error']}")
            
            # Print choice if successful
            if result['success'] and result['choice']:
                winner_subject = result['winner_subject'] or 'Unknown'
                print(f"  Choice: {result['choice']} (Winner: {winner_subject})")
        
        # Save results
        self.results = results
        self._save_results(experiment_path)
        
        print(f"Experiment completed: {self.experiment_metadata['successful_comparisons']}/{len(pairs)} successful")
        return experiment_path
    
    def _save_results(self, experiment_path: str):
        """Save experiment results and metadata."""
        # Save metadata
        metadata_path = os.path.join(experiment_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Save results as CSV
        results_path = os.path.join(experiment_path, 'results.csv')
        df = pd.DataFrame(self.results)
        df.to_csv(results_path, index=False)
        
        print(f"Results saved to {experiment_path}")
        print(f"  - metadata.json: Experiment parameters")
        print(f"  - results.csv: {len(self.results)} comparison results")