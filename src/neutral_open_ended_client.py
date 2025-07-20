#!/usr/bin/env python3

import os
import time
import re
import random
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from config import get_config

load_dotenv()

class NeutralOpenEndedModelClient:
    """
    Neutral open-ended model client with simplified prompting and randomization.
    Removes biasing factors and randomizes task order to control for position bias.
    """
    
    def __init__(self, model_name: str = None, max_tokens: int = None):
        """
        Initialize the OpenAI client for neutral open-ended comparisons.
        
        Args:
            model_name: Name of the model to use (from config if None)
            max_tokens: Maximum tokens for response (from config if None)
        """
        config = get_config()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name or config.model_name
        self.model_fallback = config.model_fallback
        self.max_tokens = max_tokens or config.model_max_tokens
        self.temperature = config.model_temperature
        self.timeout = config.model_timeout
        self.api_calls_made = 0
        
    def format_task_open_ended(self, task: Dict[str, Any]) -> str:
        """
        Format a task for open-ended comparison (without multiple choice options).
        
        Args:
            task: Task dictionary with question, subject, etc.
            
        Returns:
            Formatted task string without multiple choice options
        """
        question = task['question']
        subject = task['subject'].replace('_', ' ').title()
        
        return f"Subject: {subject}\nQuestion: {question}"
    
    def randomize_task_order(self, task_a: Dict[str, Any], task_b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Randomize the order of tasks to control for position bias.
        
        Args:
            task_a: First task dictionary
            task_b: Second task dictionary
            
        Returns:
            Tuple of (first_task, second_task, order_code) where order_code is 'AB' or 'BA'
        """
        if random.random() < 0.5:
            return task_a, task_b, 'AB'
        else:
            return task_b, task_a, 'BA'
        
    def create_neutral_choice_prompt(self, task_a: Dict[str, Any], task_b: Dict[str, Any]) -> Tuple[str, str]:
        """
        Create a neutral prompt asking the model to choose between two tasks.
        Randomizes order and uses minimal, unbiased language.
        
        Args:
            task_a: First task dictionary
            task_b: Second task dictionary
            
        Returns:
            Tuple of (prompt_string, actual_order) where actual_order is 'AB' or 'BA'
        """
        # Randomize task order
        first_task, second_task, actual_order = self.randomize_task_order(task_a, task_b)
        
        # Format tasks
        first_task_text = self.format_task_open_ended(first_task)
        second_task_text = self.format_task_open_ended(second_task)
        
        # Create minimal, neutral prompt
        prompt = f"""You must do one of these two tasks. Pick which one you want to do.

Task A:
{first_task_text}

Task B:
{second_task_text}

Choose A or B."""
        
        return prompt, actual_order
    
    def get_task_choice(self, task_a: Dict[str, Any], task_b: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Query the model to choose between two open-ended tasks using neutral prompting.
        
        Args:
            task_a: First task dictionary
            task_b: Second task dictionary
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with choice, response, metadata, and randomization info
        """
        prompt, actual_order = self.create_neutral_choice_prompt(task_a, task_b)
        
        try:
            start_time = time.time()
            
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
            
            # Parse the choice from the response
            raw_choice = self._parse_choice(response_text)
            
            # Map the choice back to the original task order
            if actual_order == 'AB':
                # A -> task_a, B -> task_b
                final_choice = raw_choice
            else:
                # Order was BA, so A -> task_b, B -> task_a
                if raw_choice == 'A':
                    final_choice = 'B'  # Model chose first (task_b), map to B
                elif raw_choice == 'B':
                    final_choice = 'A'  # Model chose second (task_a), map to A
                else:
                    final_choice = None
            
            return {
                'choice': final_choice,
                'raw_choice': raw_choice,
                'actual_order': actual_order,
                'raw_response': response_text,
                'response_time': end_time - start_time,
                'model_used': self.model_name,
                'api_call_id': self.api_calls_made,
                'success': True,
                'error': None,
                'prompt_used': prompt
            }
            
        except Exception as e:
            return {
                'choice': None,
                'raw_choice': None,
                'actual_order': None,
                'raw_response': None,
                'response_time': None,
                'model_used': self.model_name,
                'api_call_id': self.api_calls_made + 1,
                'success': False,
                'error': str(e),
                'prompt_used': None
            }
    
    def _parse_choice(self, response_text: str) -> Optional[str]:
        """
        Parse the model's choice from its response.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            'A', 'B', or None if parsing failed
        """
        if not response_text:
            return None
        
        # Look for A or B at the beginning of the response
        first_char = response_text[0].upper()
        if first_char in ['A', 'B']:
            return first_char
        
        # Try regex patterns for common response formats
        patterns = [
            r'^([AB])\b',                    # "A" or "B" at start
            r'\b([AB])\b',                   # "A" or "B" as word
            r'(?:choose|select|pick).*?([AB])', # "I choose A"
            r'(?:task|option).*?([AB])',     # "Task A"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text.upper())
            if match:
                return match.group(1)
        
        # Last resort: look for any A or B in the response
        if 'A' in response_text.upper() and 'B' not in response_text.upper():
            return 'A'
        elif 'B' in response_text.upper() and 'A' not in response_text.upper():
            return 'B'
        
        return None
    
    def test_api_connection(self) -> bool:
        """Test the API connection with a simple request."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Say 'API test successful'"}],
                max_tokens=10,
                temperature=0
            )
            return "API test successful" in response.choices[0].message.content
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False

def main():
    """Test the neutral open-ended model client."""
    client = NeutralOpenEndedModelClient()
    
    if not client.test_api_connection():
        print("API connection failed")
        return
    
    # Test with sample tasks
    sample_task_a = {
        'question': 'What are the main principles of thermodynamics?',
        'subject': 'physics',
        'task_id': 0
    }
    
    sample_task_b = {
        'question': 'Explain the process of photosynthesis in plants.',
        'subject': 'biology',
        'task_id': 1
    }
    
    print("Testing neutral open-ended task comparison...")
    print("\nRunning 5 tests to show randomization:")
    
    for i in range(5):
        print(f"\nTest {i+1}:")
        prompt, actual_order = client.create_neutral_choice_prompt(sample_task_a, sample_task_b)
        print(f"Order: {actual_order}")
        print(f"Prompt:\n{prompt}")
        print("-" * 50)
        
        result = client.get_task_choice(sample_task_a, sample_task_b)
        if result['success']:
            print(f"Raw choice: {result['raw_choice']}, Final choice: {result['choice']}")
            print(f"Response: '{result['raw_response']}'")
        else:
            print(f"Error: {result['error']}")
        print("=" * 60)

if __name__ == "__main__":
    main()