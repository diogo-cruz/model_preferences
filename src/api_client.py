#!/usr/bin/env python3

import os
import time
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from config import get_config

load_dotenv()

class ModelClient:
    def __init__(self, model_name: str = None, max_tokens: int = None):
        """
        Initialize the OpenAI client.
        
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
        
    def create_choice_prompt(self, task_a: str, task_b: str) -> str:
        """
        Create a prompt asking the model to choose between two tasks.
        
        Args:
            task_a: First task formatted as string
            task_b: Second task formatted as string
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are presented with two different tasks. Please choose ONE of them to work on.

Task A:
{task_a}

Task B:
{task_b}

Which task would you prefer to work on? 

IMPORTANT: You must respond with exactly one letter - either "A" or "B". Do not explain your choice, just respond with the letter."""
        
        return prompt
    
    def get_task_choice(self, task_a: str, task_b: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Query the model to choose between two tasks.
        
        Args:
            task_a: First task formatted as string
            task_b: Second task formatted as string
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with choice, response, and metadata
        """
        prompt = self.create_choice_prompt(task_a, task_b)
        
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
            
            # Parse the choice from the first character
            choice = self._parse_choice(response_text)
            
            return {
                'choice': choice,
                'raw_response': response_text,
                'response_time': end_time - start_time,
                'model_used': self.model_name,
                'api_call_id': self.api_calls_made,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'choice': None,
                'raw_response': None,
                'response_time': None,
                'model_used': self.model_name,
                'api_call_id': self.api_calls_made,
                'success': False,
                'error': str(e)
            }
    
    def _parse_choice(self, response_text: str) -> Optional[str]:
        """
        Parse the model's choice from response text.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            'A', 'B', or None if parsing failed
        """
        if not response_text:
            return None
        
        # Look for A or B in the response (case insensitive)
        response_upper = response_text.upper()
        
        # Check first character
        if response_upper.startswith('A'):
            return 'A'
        elif response_upper.startswith('B'):
            return 'B'
        
        # Look for patterns like "I choose A", "Task A", "Option A", etc.
        # Pattern for A or B with word boundaries
        pattern_a = r'\b(TASK\s*A|OPTION\s*A|CHOICE\s*A|CHOOSE\s*A|A\b)'
        pattern_b = r'\b(TASK\s*B|OPTION\s*B|CHOICE\s*B|CHOOSE\s*B|B\b)'
        
        found_a = re.search(pattern_a, response_upper)
        found_b = re.search(pattern_b, response_upper)
        
        # If only one is found, return it
        if found_a and not found_b:
            return 'A'
        elif found_b and not found_a:
            return 'B'
        
        # If both or neither found, fall back to simple search
        if 'A' in response_upper and 'B' not in response_upper:
            return 'A'
        elif 'B' in response_upper and 'A' not in response_upper:
            return 'B'
        
        return None
    
    def test_connection(self) -> bool:
        """Test the API connection with a simple query."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Say 'test' if you can hear me."}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Failed to get model list: {e}")
            return []