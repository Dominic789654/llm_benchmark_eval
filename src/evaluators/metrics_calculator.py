"""
Metrics calculation utilities for benchmark evaluation.
"""
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict
from .base_evaluator import EvaluationResult


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Calculate pass@k metric.
        
        Args:
            n: Total number of samples
            c: Number of correct samples
            k: k in pass@k
        
        Returns:
            Pass@k value
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)
    
    @staticmethod
    def calculate_accuracy(results: List[EvaluationResult]) -> float:
        """
        Calculate overall accuracy.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Accuracy as a float between 0 and 1
        """
        if not results:
            return 0.0
        
        correct = sum(1 for r in results if r.is_correct)
        return correct / len(results)
    
    @staticmethod
    def calculate_pass_at_k_metrics(
        problem_results: Dict[str, List[EvaluationResult]], 
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[int, float]:
        """
        Calculate pass@k metrics for multiple k values.
        
        Args:
            problem_results: Dictionary mapping problem IDs to lists of results
            k_values: List of k values to calculate
        
        Returns:
            Dictionary mapping k values to pass@k scores
        """
        pass_at_k_scores = {}
        
        for k in k_values:
            total_pass_at_k = 0.0
            valid_problems = 0
            
            for problem_id, results in problem_results.items():
                if not results:
                    continue
                
                n = len(results)
                c = sum(1 for r in results if r.is_correct)
                
                if n >= k:  # Only calculate if we have enough samples
                    total_pass_at_k += MetricsCalculator.pass_at_k(n, c, k)
                    valid_problems += 1
            
            if valid_problems > 0:
                pass_at_k_scores[k] = total_pass_at_k / valid_problems
            else:
                pass_at_k_scores[k] = 0.0
        
        return pass_at_k_scores
    
    @staticmethod
    def calculate_token_efficiency(
        results: List[EvaluationResult],
        token_counts: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Calculate token efficiency metrics.
        
        Args:
            results: List of evaluation results
            token_counts: Optional list of token counts for each result
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not results:
            return {"avg_tokens_per_sample": 0.0, "avg_tokens_per_correct": 0.0}
        
        # Extract token counts from metadata if not provided
        if token_counts is None:
            token_counts = []
            for result in results:
                tokens = 0
                if result.metadata and "tokens_used" in result.metadata:
                    tokens = result.metadata["tokens_used"]
                token_counts.append(tokens)
        
        total_tokens = sum(token_counts)
        correct_results = [r for r in results if r.is_correct]
        correct_tokens = [token_counts[i] for i, r in enumerate(results) if r.is_correct]
        
        metrics = {
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(results) if results else 0.0,
            "avg_tokens_per_correct": sum(correct_tokens) / len(correct_tokens) if correct_tokens else 0.0,
            "token_efficiency": len(correct_results) / total_tokens if total_tokens > 0 else 0.0
        }
        
        return metrics
    
    @staticmethod
    def calculate_comprehensive_metrics(
        results: List[EvaluationResult],
        problem_groups: Optional[Dict[str, List[str]]] = None,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            results: List of evaluation results
            problem_groups: Optional grouping of problems (e.g., by difficulty)
            k_values: List of k values for pass@k calculation
        
        Returns:
            Comprehensive metrics dictionary
        """
        if not results:
            return {"error": "No results provided"}
        
        # Group results by problem ID
        problem_results = defaultdict(list)
        for result in results:
            # Extract problem ID from sample_id (assuming format: dataset_split_idx)
            problem_id = result.sample_id
            problem_results[problem_id].append(result)
        
        # Basic metrics
        accuracy = MetricsCalculator.calculate_accuracy(results)
        pass_at_k_scores = MetricsCalculator.calculate_pass_at_k_metrics(
            dict(problem_results), k_values
        )
        
        # Token efficiency
        token_metrics = MetricsCalculator.calculate_token_efficiency(results)
        
        # Group-specific metrics if provided
        group_metrics = {}
        if problem_groups:
            for group_name, problem_ids in problem_groups.items():
                group_results = [r for r in results if r.sample_id in problem_ids]
                if group_results:
                    group_metrics[group_name] = {
                        "accuracy": MetricsCalculator.calculate_accuracy(group_results),
                        "count": len(group_results),
                        "correct_count": sum(1 for r in group_results if r.is_correct)
                    }
        
        # Confidence analysis (if available)
        confidence_metrics = {}
        confidence_values = [r.confidence for r in results if r.confidence is not None]
        if confidence_values:
            confidence_metrics = {
                "avg_confidence": sum(confidence_values) / len(confidence_values),
                "min_confidence": min(confidence_values),
                "max_confidence": max(confidence_values)
            }
        
        return {
            "overall": {
                "total_samples": len(results),
                "correct_samples": sum(1 for r in results if r.is_correct),
                "accuracy": accuracy,
                "pass_at_k": pass_at_k_scores
            },
            "token_efficiency": token_metrics,
            "group_metrics": group_metrics,
            "confidence_metrics": confidence_metrics,
            "problem_count": len(problem_results)
        }
