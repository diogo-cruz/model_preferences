#!/usr/bin/env python3

import os
import time
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from config import get_config

load_dotenv()

class OpenEndedModelClient:
    """
    Model client for open-ended task comparisons.
    Presents tasks without multiple choice options.
    """
    
    def __init__(self, model_name: str = None, max_tokens: int = None):
        """
        Initialize the OpenAI client for open-ended comparisons.
        
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
        
    def create_open_ended_choice_prompt(self, task_a: Dict[str, Any], task_b: Dict[str, Any]) -> str:
        """
        Create a prompt asking the model to choose between two open-ended tasks.
        
        Args:
            task_a: First task dictionary
            task_b: Second task dictionary
            
        Returns:
            Formatted prompt string
        """
        task_a_text = self.format_task_open_ended(task_a)
        task_b_text = self.format_task_open_ended(task_b)
        
        prompt = f"""You are presented with two different open-ended questions. Please choose ONE that you would prefer to work on.

Task A:
{task_a_text}

Task B:
{task_b_text}

Which question would you prefer to answer? Consider factors like:
- Your expertise and knowledge in the subject area
- The clarity and specificity of the question
- Your interest in the topic
- The potential for providing a comprehensive answer

IMPORTANT: You must respond with exactly one letter - either "A" or "B". Do not explain your choice, just respond with the letter."""
        
        return prompt
    
    def get_task_choice(self, task_a: Dict[str, Any], task_b: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Query the model to choose between two open-ended tasks.
        
        Args:
            task_a: First task dictionary
            task_b: Second task dictionary
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with choice, response, and metadata
        """
        prompt = self.create_open_ended_choice_prompt(task_a, task_b)
        
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
                'api_call_id': self.api_calls_made + 1,
                'success': False,
                'error': str(e)
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
    """Test the open-ended model client."""
    client = OpenEndedModelClient()
    
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
    
    print("Testing open-ended task comparison...")
    print("\nPrompt:")
    print(client.create_open_ended_choice_prompt(sample_task_a, sample_task_b))
    
    result = client.get_task_choice(sample_task_a, sample_task_b)
    print(f"\nResult: {result}")
    
    if result['success']:
        print(f"Model chose: {result['choice']}")
        print(f"Raw response: '{result['raw_response']}'")
        print(f"Response time: {result['response_time']:.3f}s")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()