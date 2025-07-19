#!/usr/bin/env python3

import random
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from api_client import ModelClient
from config import get_config

class BiasCorrectionStrategy:
    """Strategies for correcting position bias in pairwise comparisons."""
    
    def __init__(self):
        self.config = get_config()
    
    def create_randomized_prompt(self, task_a: str, task_b: str, randomize_labels: bool = True) -> Tuple[str, Dict[str, str]]:
        """
        Create a prompt with randomized task labels to reduce position bias.
        
        Args:
            task_a: First task formatted as string
            task_b: Second task formatted as string
            randomize_labels: If True, randomly assign which task gets label 1 vs 2
            
        Returns:
            Tuple of (prompt, label_mapping) where label_mapping shows which original task maps to which label
        """
        # Randomly assign task order and labels
        if randomize_labels and random.random() < 0.5:
            # Swap tasks
            first_task = task_b
            second_task = task_a
            label_mapping = {'1': 'B', '2': 'A'}  # Label 1 = original task B, Label 2 = original task A
        else:
            first_task = task_a
            second_task = task_b
            label_mapping = {'1': 'A', '2': 'B'}  # Label 1 = original task A, Label 2 = original task B
        
        prompt = f"""You are presented with two different academic tasks. Please choose ONE of them to work on.

Task 1:
{first_task}

Task 2:
{second_task}

Which task would you prefer to work on?

IMPORTANT: You must respond with exactly one number - either "1" or "2". Do not explain your choice, just respond with the number."""
        
        return prompt, label_mapping
    
    def create_neutral_ordering_prompt(self, task_a: str, task_b: str) -> str:
        """
        Create a prompt that presents tasks in a more neutral way.
        
        Args:
            task_a: First task formatted as string
            task_b: Second task formatted as string
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Below are two academic tasks. Please select the one you would prefer to work on.

OPTION 1:
{task_a}

OPTION 2:
{task_b}

Please indicate your preference by responding with "OPTION 1" or "OPTION 2"."""
        
        return prompt
    
    def create_blind_comparison_prompt(self, task_a: str, task_b: str) -> Tuple[str, Dict[str, str]]:
        """
        Create a prompt that presents tasks without explicit ordering cues.
        
        Args:
            task_a: First task formatted as string
            task_b: Second task formatted as string
            
        Returns:
            Tuple of (prompt, choice_mapping)
        """
        # Use more neutral identifiers
        identifiers = ['ALPHA', 'BETA'] if random.random() < 0.5 else ['BETA', 'ALPHA']
        
        if identifiers[0] == 'ALPHA':
            choice_mapping = {'ALPHA': 'A', 'BETA': 'B'}
            first_task, second_task = task_a, task_b
        else:
            choice_mapping = {'ALPHA': 'B', 'BETA': 'A'}
            first_task, second_task = task_b, task_a
        
        prompt = f"""Consider these two academic tasks:

{identifiers[0]}:
{first_task}

{identifiers[1]}:
{second_task}

Which would you choose to work on? Respond with "{identifiers[0]}" or "{identifiers[1]}"."""
        
        return prompt, choice_mapping

class BiasAwareClient(ModelClient):
    """Extended ModelClient with bias correction capabilities."""
    
    def __init__(self, model_name: str = None, max_tokens: int = None, bias_strategy: str = "randomized"):
        """
        Initialize bias-aware client.
        
        Args:
            model_name: Name of the model to use
            max_tokens: Maximum tokens for response
            bias_strategy: Strategy for bias correction ("randomized", "neutral", "blind")
        """
        super().__init__(model_name, max_tokens)
        self.bias_strategy = bias_strategy
        self.bias_corrector = BiasCorrectionStrategy()
        
    def get_bias_corrected_choice(self, task_a: str, task_b: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Query the model with bias correction strategy.
        
        Args:
            task_a: First task formatted as string
            task_b: Second task formatted as string
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with choice, response, and metadata including bias correction info
        """
        start_time = time.time()
        
        try:
            # Apply bias correction strategy
            if self.bias_strategy == "randomized":
                prompt, label_mapping = self.bias_corrector.create_randomized_prompt(task_a, task_b)
                expected_responses = ['1', '2']
            elif self.bias_strategy == "neutral":
                prompt = self.bias_corrector.create_neutral_ordering_prompt(task_a, task_b)
                label_mapping = {'OPTION 1': 'A', 'OPTION 2': 'B'}
                expected_responses = ['OPTION 1', 'OPTION 2']
            elif self.bias_strategy == "blind":
                prompt, label_mapping = self.bias_corrector.create_blind_comparison_prompt(task_a, task_b)
                expected_responses = list(label_mapping.keys())
            else:
                # Fallback to original method
                prompt = self.create_choice_prompt(task_a, task_b)
                label_mapping = {'A': 'A', 'B': 'B'}
                expected_responses = ['A', 'B']
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=timeout
            )
            
            end_time = time.time()
            self.api_calls_made += 1
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse the choice with bias correction mapping
            raw_choice = self._parse_bias_corrected_choice(response_text, expected_responses)
            
            # Map back to original A/B choice
            if raw_choice and raw_choice in label_mapping:
                final_choice = label_mapping[raw_choice]
            else:
                final_choice = None
            
            return {
                'choice': final_choice,
                'raw_choice': raw_choice,
                'raw_response': response_text,
                'response_time': end_time - start_time,
                'model_used': self.model_name,
                'api_call_id': self.api_calls_made,
                'bias_strategy': self.bias_strategy,
                'label_mapping': label_mapping,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'choice': None,
                'raw_choice': None,
                'raw_response': None,
                'response_time': None,
                'model_used': self.model_name,
                'api_call_id': self.api_calls_made,
                'bias_strategy': self.bias_strategy,
                'label_mapping': {},
                'success': False,
                'error': str(e)
            }
    
    def _parse_bias_corrected_choice(self, response_text: str, expected_responses: List[str]) -> str:
        """
        Parse choice from bias-corrected response.
        
        Args:
            response_text: Raw response from the model
            expected_responses: List of valid response options
            
        Returns:
            Parsed choice or None if parsing failed
        """
        if not response_text:
            return None
        
        response_upper = response_text.upper().strip()
        
        # Look for exact matches first
        for expected in expected_responses:
            if expected.upper() in response_upper:
                return expected
        
        # For numbered responses, also check for just the digit
        for expected in expected_responses:
            if expected.isdigit() and expected in response_upper:
                return expected
        
        return None

def test_bias_correction_strategies():
    """Test different bias correction strategies."""
    import time
    
    print("=" * 60)
    print("TESTING BIAS CORRECTION STRATEGIES")
    print("=" * 60)
    
    # Sample tasks for testing
    task_a = """Subject: Mathematics
Question: What is the derivative of x^2?
Choices:
A) 2x
B) x^2
C) 2
D) x"""
    
    task_b = """Subject: History
Question: When did World War II end?
Choices:
A) 1944
B) 1945
C) 1946
D) 1947"""
    
    strategies = ["randomized", "neutral", "blind"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        client = BiasAwareClient(bias_strategy=strategy)
        
        try:
            result = client.get_bias_corrected_choice(task_a, task_b)
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Raw choice: {result['raw_choice']}")
                print(f"  Final choice: {result['choice']}")
                print(f"  Label mapping: {result['label_mapping']}")
                print(f"  Response time: {result['response_time']:.2f}s")
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    test_bias_correction_strategies()