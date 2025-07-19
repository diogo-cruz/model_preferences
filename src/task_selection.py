#!/usr/bin/env python3

import random
from datasets import load_dataset
from typing import List, Dict, Any
from config import get_config

class TaskSelector:
    def __init__(self, seed: int = None):
        config = get_config()
        self.seed = seed or config.experiment_seed
        self.dataset = None
        self.selected_tasks = []
    
    def load_mmlu_dataset(self):
        """Load the MMLU dataset."""
        print("Loading MMLU dataset...")
        self.dataset = load_dataset("cais/mmlu", "all")
        print(f"Dataset loaded with {len(self.dataset['test'])} test samples")
    
    def get_unique_subjects(self) -> List[str]:
        """Get list of unique subjects from the dataset."""
        if not self.dataset:
            self.load_mmlu_dataset()
        
        subjects = set()
        for item in self.dataset['test']:
            if item['subject']:  # Filter out empty subjects
                subjects.add(item['subject'])
        return sorted(list(subjects))
    
    def sample_tasks(self, n_tasks: int = 100) -> List[Dict[str, Any]]:
        """
        Sample n_tasks from MMLU dataset with balanced subject representation.
        Uses fixed seed for reproducibility.
        """
        if not self.dataset:
            self.load_mmlu_dataset()
        
        random.seed(self.seed)
        
        # Get all valid tasks (exclude empty subjects)
        valid_tasks = [item for item in self.dataset['test'] if item['subject']]
        
        # Group tasks by subject for balanced sampling
        tasks_by_subject = {}
        for task in valid_tasks:
            subject = task['subject']
            if subject not in tasks_by_subject:
                tasks_by_subject[subject] = []
            tasks_by_subject[subject].append(task)
        
        subjects = list(tasks_by_subject.keys())
        print(f"Found {len(subjects)} unique subjects")
        
        # Sample tasks with balanced representation across subjects
        selected_tasks = []
        tasks_per_subject = n_tasks // len(subjects)
        remaining_tasks = n_tasks % len(subjects)
        
        for i, subject in enumerate(subjects):
            # Some subjects get one extra task to reach exactly n_tasks
            n_from_subject = tasks_per_subject + (1 if i < remaining_tasks else 0)
            
            subject_tasks = tasks_by_subject[subject]
            if len(subject_tasks) >= n_from_subject:
                sampled = random.sample(subject_tasks, n_from_subject)
            else:
                # If subject has fewer tasks than needed, take all
                sampled = subject_tasks
            
            selected_tasks.extend(sampled)
        
        # If we still need more tasks, sample randomly from remaining
        if len(selected_tasks) < n_tasks:
            remaining_needed = n_tasks - len(selected_tasks)
            all_remaining = [task for task in valid_tasks if task not in selected_tasks]
            additional = random.sample(all_remaining, min(remaining_needed, len(all_remaining)))
            selected_tasks.extend(additional)
        
        # Trim to exact number if we somehow got too many
        selected_tasks = selected_tasks[:n_tasks]
        
        # Add unique IDs for tracking
        for i, task in enumerate(selected_tasks):
            task['task_id'] = i
        
        self.selected_tasks = selected_tasks
        print(f"Selected {len(selected_tasks)} tasks from {len(set(task['subject'] for task in selected_tasks))} subjects")
        
        return selected_tasks
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary statistics of selected tasks."""
        if not self.selected_tasks:
            return {}
        
        subject_counts = {}
        for task in self.selected_tasks:
            subject = task['subject']
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        return {
            'total_tasks': len(self.selected_tasks),
            'unique_subjects': len(subject_counts),
            'subject_distribution': subject_counts,
            'seed_used': self.seed
        }
    
    def format_task_for_prompt(self, task: Dict[str, Any]) -> str:
        """Format a task for use in model prompts."""
        question = task['question']
        choices = task['choices']
        subject = task['subject'].replace('_', ' ').title()
        
        choices_text = '\n'.join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        
        return f"Subject: {subject}\nQuestion: {question}\nChoices:\n{choices_text}"