"""
Metrics calculation utilities.
"""
import math
from typing import List, Optional


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k
    
    Returns:
        Pass@k value between 0 and 1
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def calculate_accuracy(predictions: List[bool]) -> float:
    """
    Calculate accuracy from a list of boolean predictions.
    
    Args:
        predictions: List of boolean values (True for correct, False for incorrect)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    if not predictions:
        return 0.0
    return sum(predictions) / len(predictions)


def token_efficiency(correct_count: int, total_tokens: int) -> float:
    """
    Calculate token efficiency (correct answers per token).
    
    Args:
        correct_count: Number of correct answers
        total_tokens: Total number of tokens used
    
    Returns:
        Efficiency score (correct answers per token)
    """
    if total_tokens == 0:
        return 0.0
    return correct_count / total_tokens


def calculate_confidence_metrics(confidences: List[float]) -> dict:
    """
    Calculate confidence-related metrics.
    
    Args:
        confidences: List of confidence scores
    
    Returns:
        Dictionary with confidence metrics
    """
    if not confidences:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)
    
    # Calculate standard deviation
    variance = sum((x - avg_conf) ** 2 for x in confidences) / len(confidences)
    std_conf = math.sqrt(variance)
    
    return {
        "avg": avg_conf,
        "min": min_conf,
        "max": max_conf,
        "std": std_conf
    }


def calculate_percentiles(values: List[float], percentiles: List[int] = [25, 50, 75, 90, 95]) -> dict:
    """
    Calculate percentiles for a list of values.
    
    Args:
        values: List of numerical values
        percentiles: List of percentile values to calculate
    
    Returns:
        Dictionary mapping percentile to value
    """
    if not values:
        return {p: 0.0 for p in percentiles}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    result = {}
    for p in percentiles:
        if p == 0:
            result[p] = sorted_values[0]
        elif p == 100:
            result[p] = sorted_values[-1]
        else:
            index = (p / 100) * (n - 1)
            if index.is_integer():
                result[p] = sorted_values[int(index)]
            else:
                lower = sorted_values[int(index)]
                upper = sorted_values[int(index) + 1]
                result[p] = lower + (upper - lower) * (index - int(index))
    
    return result
