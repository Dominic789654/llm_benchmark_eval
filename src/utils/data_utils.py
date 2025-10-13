"""
Data utilities for loading and processing benchmark data.
"""
import json
import re
from typing import List, Dict, Any, Optional


def load_jsonl(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        max_samples: Maximum number of samples to load
    
    Returns:
        List of loaded samples
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_answer(text: str) -> str:
    """
    Extract the final answer from text using common patterns.
    
    Args:
        text: Text to extract answer from
    
    Returns:
        Extracted answer
    """
    # Look for boxed answer first
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for final answer patterns
    answer_patterns = [
        r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
        r'[Ff]inal answer:?\s*(.+?)(?:\n|$)',
        r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
        r'[Tt]herefore,?\s*(.+?)(?:\n|$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # Fallback: return the text as is (cleaned)
    return text.strip()


def extract_gsm8k_answer(answer_str: str) -> str:
    """
    Extract numerical answer from GSM8K format.
    
    Args:
        answer_str: GSM8K answer string
    
    Returns:
        Extracted numerical answer
    """
    if not isinstance(answer_str, str):
        return str(answer_str)
    
    # GSM8K format: "explanation\n#### final_answer"
    if '####' in answer_str:
        return answer_str.split('####')[-1].strip().replace(',', '')
    
    # Fallback: try to extract the last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str)
    if numbers:
        return numbers[-1]
    
    return answer_str.strip()


def create_xverify_prompt(question: str, model_output: str, correct_answer: str) -> str:
    """
    Create a prompt for xVerify evaluation.
    
    Args:
        question: The original question
        model_output: Model's generated output
        correct_answer: The correct answer
    
    Returns:
        Formatted xVerify prompt
    """
    # This is a simplified version - the actual xVerify prompt template 
    # should be imported from the xVerify library
    prompt = f"""Please evaluate whether the following answer is correct.

Question: {question}

Model Output: {model_output}

Correct Answer: {correct_answer}

Is the model output correct? Please respond with "Correct" or "Incorrect" and provide a brief explanation."""
    
    return prompt


def truncate_to_max_tokens(text: str, max_tokens: int, tokenizer=None) -> str:
    """
    Truncate text to maximum number of tokens.
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        tokenizer: Tokenizer to use for token counting (optional)
    
    Returns:
        Truncated text
    """
    if max_tokens <= 0:
        return text
    
    if tokenizer is not None:
        # Use actual tokenizer for precise truncation
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    else:
        # Rough approximation: 4 characters per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]


def standardize_sample_format(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Standardize a sample to the common format.
    
    Args:
        sample: Raw sample from dataset
        dataset_name: Name of the source dataset
    
    Returns:
        Standardized sample format
    """
    # Map common field variations to standard names
    field_mappings = {
        'question': ['question', 'problem', 'input', 'query'],
        'answer': ['answer', 'solution', 'target', 'output'],
        'id': ['id', 'idx', 'index', 'problem_id']
    }
    
    standardized = {'dataset': dataset_name}
    
    for standard_field, possible_fields in field_mappings.items():
        for field in possible_fields:
            if field in sample:
                standardized[standard_field] = sample[field]
                break
        
        # Set default if not found
        if standard_field not in standardized:
            standardized[standard_field] = ""
    
    # Copy any additional metadata
    standardized['metadata'] = {
        k: v for k, v in sample.items() 
        if k not in ['question', 'problem', 'input', 'query', 'answer', 'solution', 'target', 'output', 'id', 'idx', 'index', 'problem_id']
    }
    
    return standardized


def validate_sample(sample: Dict[str, Any]) -> bool:
    """
    Validate that a sample has required fields.
    
    Args:
        sample: Sample to validate
    
    Returns:
        True if sample is valid, False otherwise
    """
    required_fields = ['question', 'answer']
    return all(field in sample and sample[field] for field in required_fields)
