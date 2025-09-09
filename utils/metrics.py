"""
Evaluation Metrics for EEG to Text Translation

This module provides various evaluation metrics for assessing the performance
of EEG to text translation models.
"""

import numpy as np
from typing import List, Dict, Union
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


def compute_bleu_score(predictions: List[str], targets: List[str]) -> float:
    """
    Compute BLEU score for text generation.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        
    Returns:
        Average BLEU score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    for pred, target in zip(predictions, targets):
        # Tokenize
        pred_tokens = pred.lower().split()
        target_tokens = target.lower().split()
        
        # Compute BLEU score
        score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)


def compute_rouge_scores(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, target in zip(predictions, targets):
        scores = scorer.score(target, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }


def compute_exact_match_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        
    Returns:
        Exact match accuracy
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.lower().strip() == target.lower().strip():
            correct += 1
    
    return correct / len(predictions)


def compute_word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute word-level accuracy.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        
    Returns:
        Word-level accuracy
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    total_words = 0
    correct_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.lower().split()
        target_words = target.lower().split()
        
        total_words += len(target_words)
        
        # Count correct words
        for word in pred_words:
            if word in target_words:
                correct_words += 1
    
    return correct_words / total_words if total_words > 0 else 0.0


def compute_character_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute character-level accuracy.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        
    Returns:
        Character-level accuracy
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        pred_chars = list(pred.lower())
        target_chars = list(target.lower())
        
        total_chars += len(target_chars)
        
        # Count correct characters
        for i, char in enumerate(pred_chars):
            if i < len(target_chars) and char == target_chars[i]:
                correct_chars += 1
    
    return correct_chars / total_chars if total_chars > 0 else 0.0


def compute_meteor_score(predictions: List[str], targets: List[str]) -> float:
    """
    Compute METEOR score (simplified version).
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        
    Returns:
        METEOR score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # This is a simplified METEOR implementation
    # For full METEOR, you would need additional dependencies
    meteor_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        
        if len(target_words) == 0:
            meteor_scores.append(0.0)
            continue
        
        # Precision and recall
        matches = len(pred_words.intersection(target_words))
        precision = matches / len(pred_words) if len(pred_words) > 0 else 0.0
        recall = matches / len(target_words)
        
        # F-score
        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = 2 * precision * recall / (precision + recall)
        
        meteor_scores.append(f_score)
    
    return np.mean(meteor_scores)


def compute_metrics(predictions: List[str], 
                   targets: List[str], 
                   metrics: List[str]) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        metrics: List of metric names to compute
        
    Returns:
        Dictionary with computed metrics
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    results = {}
    
    for metric in metrics:
        try:
            if metric == 'bleu':
                results['bleu'] = compute_bleu_score(predictions, targets)
            elif metric == 'rouge':
                rouge_scores = compute_rouge_scores(predictions, targets)
                results.update(rouge_scores)
            elif metric == 'accuracy':
                results['accuracy'] = compute_exact_match_accuracy(predictions, targets)
            elif metric == 'word_accuracy':
                results['word_accuracy'] = compute_word_accuracy(predictions, targets)
            elif metric == 'char_accuracy':
                results['char_accuracy'] = compute_character_accuracy(predictions, targets)
            elif metric == 'meteor':
                results['meteor'] = compute_meteor_score(predictions, targets)
            else:
                logger.warning(f"Unknown metric: {metric}")
        except Exception as e:
            logger.error(f"Error computing metric {metric}: {e}")
            results[metric] = 0.0
    
    return results


def print_metrics_summary(metrics: Dict[str, float], model_name: str = ""):
    """
    Print a summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model for display
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:15s}: {value:.4f}")
        else:
            print(f"{metric:15s}: {value}")
    
    print(f"{'='*50}")


def save_metrics_to_file(metrics: Dict[str, float], 
                        filepath: str, 
                        model_name: str = ""):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
        model_name: Name of the model
    """
    import json
    
    output_data = {
        'model_name': model_name,
        'metrics': metrics,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Metrics saved to {filepath}")


def compare_models(model_metrics: Dict[str, Dict[str, float]], 
                  output_file: str = None) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models and their metrics.
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        output_file: Optional file to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # Get all unique metrics
    all_metrics = set()
    for metrics in model_metrics.values():
        all_metrics.update(metrics.keys())
    
    # Create comparison table
    for metric in sorted(all_metrics):
        comparison[metric] = {}
        for model_name, metrics in model_metrics.items():
            comparison[metric][model_name] = metrics.get(metric, 0.0)
    
    # Find best model for each metric
    best_models = {}
    for metric, model_scores in comparison.items():
        best_model = max(model_scores.items(), key=lambda x: x[1])
        best_models[metric] = best_model
    
    comparison['best_models'] = best_models
    
    # Print comparison
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Header
    model_names = list(model_metrics.keys())
    header = f"{'Metric':<20}"
    for model_name in model_names:
        header += f"{model_name:<15}"
    print(header)
    print("-" * 80)
    
    # Metrics
    for metric in sorted(all_metrics):
        if metric == 'best_models':
            continue
        row = f"{metric:<20}"
        for model_name in model_names:
            value = comparison[metric].get(model_name, 0.0)
            row += f"{value:<15.4f}"
        print(row)
    
    # Best models summary
    print(f"\n{'='*80}")
    print("BEST MODELS BY METRIC")
    print(f"{'='*80}")
    for metric, (model_name, score) in best_models.items():
        print(f"{metric:<20}: {model_name} ({score:.4f})")
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Model comparison saved to {output_file}")
    
    return comparison 