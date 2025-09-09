#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for EEG to Text Translation

This script evaluates both LSTM-BART and BART baseline models and generates
detailed comparison reports.
"""

import argparse
import logging
import sys
import torch
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_bart import LSTMBartModel
from models.bart_baseline import BartBaselineModel
from utils.data_loader import EEGTextDataLoader
from utils.metrics import compute_metrics, compare_models, print_metrics_summary
from utils.visualization import plot_model_comparison, plot_prediction_examples

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_config(checkpoint_path: str, model_type: str, device: str = 'cuda'):
    """
    Load model and configuration from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, tokenizer)
    """
    logger.info(f"Loading {model_type} model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    if model_type == 'lstm_bart':
        model = LSTMBartModel(
            eeg_input_size=config['model']['eeg_input_size'],
            lstm_hidden_size=config['model']['lstm_hidden_size'],
            lstm_num_layers=config['model']['lstm_num_layers'],
            bart_model_name=config['model']['bart_model_name'],
            max_length=config['model']['max_length'],
            dropout=config['model']['dropout'],
            use_features=config['model']['use_features'],
            feature_size=config['model']['feature_size']
        )
    elif model_type == 'bart_baseline':
        model = BartBaselineModel(
            eeg_input_size=config['model']['eeg_input_size'],
            bart_model_name=config['model']['bart_model_name'],
            max_length=config['model']['max_length'],
            dropout=config['model']['dropout'],
            use_features=config['model']['use_features'],
            feature_size=config['model']['feature_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create data loader to get tokenizer
    data_loader = EEGTextDataLoader(
        data_dir=config['data']['data_dir'],
        tokenizer_name=config['data']['tokenizer_name']
    )
    tokenizer = data_loader.get_tokenizer()
    
    logger.info(f"Model loaded successfully. Epoch: {checkpoint['epoch']}")
    return model, config, tokenizer


def evaluate_single_model(model: torch.nn.Module,
                         test_dataloader,
                         tokenizer,
                         model_name: str,
                         device: str) -> Dict[str, float]:
    """
    Evaluate a single model on test data.
    
    Args:
        model: Trained model
        test_dataloader: Test data loader
        tokenizer: BART tokenizer
        model_name: Name of the model
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    all_predictions = []
    all_targets = []
    all_eeg_data = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate predictions
            generated_ids = model.generate(
                eeg_data=batch['eeg_data'],
                max_length=512
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            # Get targets
            targets = tokenizer.batch_decode(
                batch['target_input_ids'], 
                skip_special_tokens=True
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_eeg_data.extend(batch['eeg_data'].cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(
        predictions=all_predictions,
        targets=all_targets,
        metrics=['bleu', 'rouge', 'accuracy', 'word_accuracy', 'char_accuracy', 'meteor']
    )
    
    # Print results
    print_metrics_summary(metrics, model_name)
    
    return metrics, all_predictions, all_targets, all_eeg_data


def comprehensive_evaluation(lstm_bart_path: str,
                           bart_baseline_path: str,
                           data_dir: str,
                           output_dir: str,
                           device: str = 'cuda') -> Dict:
    """
    Perform comprehensive evaluation of both models.
    
    Args:
        lstm_bart_path: Path to LSTM-BART checkpoint
        bart_baseline_path: Path to BART baseline checkpoint
        data_dir: Directory containing test data
        output_dir: Directory to save results
        device: Device to use
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    logger.info("Starting comprehensive evaluation...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load models
    lstm_bart_model, lstm_config, lstm_tokenizer = load_model_and_config(
        lstm_bart_path, 'lstm_bart', device
    )
    
    bart_baseline_model, bart_config, bart_tokenizer = load_model_and_config(
        bart_baseline_path, 'bart_baseline', device
    )
    
    # Create data loader
    data_loader = EEGTextDataLoader(
        data_dir=data_dir,
        batch_size=32,
        shuffle=False
    )
    
    test_dataloader = data_loader.get_dataloader('test')
    logger.info(f"Test samples: {len(test_dataloader.dataset)}")
    
    # Evaluate LSTM-BART
    lstm_metrics, lstm_predictions, lstm_targets, lstm_eeg_data = evaluate_single_model(
        lstm_bart_model, test_dataloader, lstm_tokenizer, 'LSTM-BART', device
    )
    
    # Evaluate BART baseline
    bart_metrics, bart_predictions, bart_targets, bart_eeg_data = evaluate_single_model(
        bart_baseline_model, test_dataloader, bart_tokenizer, 'BART-Baseline', device
    )
    
    # Compare models
    model_metrics = {
        'LSTM-BART': lstm_metrics,
        'BART-Baseline': bart_metrics
    }
    
    comparison_results = compare_models(
        model_metrics, 
        output_file=str(output_path / 'model_comparison.json')
    )
    
    # Generate detailed analysis
    analysis_results = generate_detailed_analysis(
        lstm_predictions, lstm_targets, lstm_eeg_data,
        bart_predictions, bart_targets, bart_eeg_data,
        output_path
    )
    
    # Create visualizations
    create_evaluation_visualizations(
        model_metrics, lstm_predictions, lstm_targets,
        bart_predictions, bart_targets, output_path
    )
    
    # Generate comprehensive report
    generate_comprehensive_report(
        model_metrics, comparison_results, analysis_results, output_path
    )
    
    logger.info(f"Comprehensive evaluation completed. Results saved to {output_path}")
    
    return {
        'model_metrics': model_metrics,
        'comparison_results': comparison_results,
        'analysis_results': analysis_results
    }


def generate_detailed_analysis(lstm_predictions: List[str],
                              lstm_targets: List[str],
                              lstm_eeg_data: List[np.ndarray],
                              bart_predictions: List[str],
                              bart_targets: List[str],
                              bart_eeg_data: List[np.ndarray],
                              output_path: Path) -> Dict:
    """
    Generate detailed analysis of model performance.
    
    Args:
        lstm_predictions: LSTM-BART predictions
        lstm_targets: LSTM-BART targets
        lstm_eeg_data: LSTM-BART EEG data
        bart_predictions: BART baseline predictions
        bart_targets: BART baseline targets
        bart_eeg_data: BART baseline EEG data
        output_path: Output directory
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Generating detailed analysis...")
    
    analysis = {}
    
    # Word length analysis
    analysis['word_length'] = analyze_word_lengths(
        lstm_predictions, lstm_targets,
        bart_predictions, bart_targets
    )
    
    # Error pattern analysis
    analysis['error_patterns'] = analyze_error_patterns(
        lstm_predictions, lstm_targets,
        bart_predictions, bart_targets
    )
    
    # EEG sequence length analysis
    analysis['eeg_analysis'] = analyze_eeg_characteristics(
        lstm_eeg_data, bart_eeg_data
    )
    
    # Save analysis
    with open(output_path / 'detailed_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    return analysis


def analyze_word_lengths(lstm_predictions: List[str],
                        lstm_targets: List[str],
                        bart_predictions: List[str],
                        bart_targets: List[str]) -> Dict:
    """Analyze performance by word length."""
    
    def get_word_length_stats(predictions, targets):
        word_lengths = [len(word.split()) for word in targets]
        correct_by_length = {}
        
        for pred, target, length in zip(predictions, targets, word_lengths):
            if length not in correct_by_length:
                correct_by_length[length] = {'correct': 0, 'total': 0}
            
            correct_by_length[length]['total'] += 1
            if pred.lower().strip() == target.lower().strip():
                correct_by_length[length]['correct'] += 1
        
        # Calculate accuracy by length
        accuracy_by_length = {}
        for length, stats in correct_by_length.items():
            accuracy_by_length[length] = stats['correct'] / stats['total']
        
        return accuracy_by_length
    
    return {
        'lstm_bart': get_word_length_stats(lstm_predictions, lstm_targets),
        'bart_baseline': get_word_length_stats(bart_predictions, bart_targets)
    }


def analyze_error_patterns(lstm_predictions: List[str],
                          lstm_targets: List[str],
                          bart_predictions: List[str],
                          bart_targets: List[str]) -> Dict:
    """Analyze common error patterns."""
    
    def get_error_patterns(predictions, targets):
        errors = []
        for pred, target in zip(predictions, targets):
            if pred.lower().strip() != target.lower().strip():
                errors.append({
                    'target': target,
                    'prediction': pred,
                    'target_length': len(target.split()),
                    'prediction_length': len(pred.split())
                })
        
        # Analyze patterns
        pattern_analysis = {
            'total_errors': len(errors),
            'length_mismatch': sum(1 for e in errors if e['target_length'] != e['prediction_length']),
            'partial_match': sum(1 for e in errors if any(word in e['target'].lower() for word in e['prediction'].lower().split())),
            'common_targets': {},
            'common_predictions': {}
        }
        
        # Count common targets and predictions
        target_counts = {}
        pred_counts = {}
        for error in errors:
            target_counts[error['target']] = target_counts.get(error['target'], 0) + 1
            pred_counts[error['prediction']] = pred_counts.get(error['prediction'], 0) + 1
        
        pattern_analysis['common_targets'] = dict(sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        pattern_analysis['common_predictions'] = dict(sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return pattern_analysis
    
    return {
        'lstm_bart': get_error_patterns(lstm_predictions, lstm_targets),
        'bart_baseline': get_error_patterns(bart_predictions, bart_targets)
    }


def analyze_eeg_characteristics(lstm_eeg_data: List[np.ndarray],
                               bart_eeg_data: List[np.ndarray]) -> Dict:
    """Analyze EEG data characteristics."""
    
    def analyze_eeg_data(eeg_data_list):
        sequence_lengths = [data.shape[0] for data in eeg_data_list]
        num_channels = eeg_data_list[0].shape[1] if eeg_data_list else 0
        
        return {
            'num_samples': len(eeg_data_list),
            'num_channels': num_channels,
            'sequence_lengths': {
                'mean': np.mean(sequence_lengths),
                'std': np.std(sequence_lengths),
                'min': np.min(sequence_lengths),
                'max': np.max(sequence_lengths)
            },
            'eeg_statistics': {
                'mean_amplitude': np.mean([np.mean(data) for data in eeg_data_list]),
                'std_amplitude': np.mean([np.std(data) for data in eeg_data_list]),
                'range_amplitude': np.mean([np.max(data) - np.min(data) for data in eeg_data_list])
            }
        }
    
    return {
        'lstm_bart_data': analyze_eeg_data(lstm_eeg_data),
        'bart_baseline_data': analyze_eeg_data(bart_eeg_data)
    }


def create_evaluation_visualizations(model_metrics: Dict,
                                   lstm_predictions: List[str],
                                   lstm_targets: List[str],
                                   bart_predictions: List[str],
                                   bart_targets: List[str],
                                   output_path: Path):
    """Create evaluation visualizations."""
    
    # Model comparison plot
    plot_model_comparison(
        metrics_file=str(output_path / 'model_comparison.json'),
        save_path=str(output_path / 'model_comparison.png'),
        show=False
    )
    
    # Prediction examples
    plot_prediction_examples(
        predictions=lstm_predictions[:10],
        targets=lstm_targets[:10],
        num_examples=10,
        save_path=str(output_path / 'lstm_bart_examples.png'),
        show=False
    )
    
    plot_prediction_examples(
        predictions=bart_predictions[:10],
        targets=bart_targets[:10],
        num_examples=10,
        save_path=str(output_path / 'bart_baseline_examples.png'),
        show=False
    )


def generate_comprehensive_report(model_metrics: Dict,
                                comparison_results: Dict,
                                analysis_results: Dict,
                                output_path: Path):
    """Generate a comprehensive evaluation report."""
    
    report = {
        'evaluation_summary': {
            'timestamp': str(np.datetime64('now')),
            'models_evaluated': list(model_metrics.keys()),
            'total_test_samples': len(analysis_results.get('eeg_analysis', {}).get('lstm_bart_data', {}).get('num_samples', 0))
        },
        'model_performance': model_metrics,
        'model_comparison': comparison_results,
        'detailed_analysis': analysis_results,
        'recommendations': generate_recommendations(model_metrics, analysis_results)
    }
    
    # Save report
    with open(output_path / 'comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    generate_markdown_report(report, output_path / 'evaluation_report.md')


def generate_recommendations(model_metrics: Dict, analysis_results: Dict) -> List[str]:
    """Generate recommendations based on evaluation results."""
    
    recommendations = []
    
    # Compare overall performance
    lstm_metrics = model_metrics.get('LSTM-BART', {})
    bart_metrics = model_metrics.get('BART-Baseline', {})
    
    if lstm_metrics.get('accuracy', 0) > bart_metrics.get('accuracy', 0):
        recommendations.append("LSTM-BART shows better overall accuracy than BART baseline")
    else:
        recommendations.append("BART baseline shows better overall accuracy than LSTM-BART")
    
    # Analyze specific metrics
    if lstm_metrics.get('bleu', 0) > bart_metrics.get('bleu', 0):
        recommendations.append("LSTM-BART achieves higher BLEU scores, indicating better text quality")
    
    if lstm_metrics.get('rouge1', 0) > bart_metrics.get('rouge1', 0):
        recommendations.append("LSTM-BART shows better ROUGE-1 scores, suggesting better word overlap")
    
    # Error pattern recommendations
    error_analysis = analysis_results.get('error_patterns', {})
    if error_analysis:
        lstm_errors = error_analysis.get('lstm_bart', {}).get('total_errors', 0)
        bart_errors = error_analysis.get('bart_baseline', {}).get('total_errors', 0)
        
        if lstm_errors < bart_errors:
            recommendations.append("LSTM-BART makes fewer total errors than BART baseline")
        else:
            recommendations.append("BART baseline makes fewer total errors than LSTM-BART")
    
    return recommendations


def generate_markdown_report(report: Dict, output_path: Path):
    """Generate a markdown version of the evaluation report."""
    
    with open(output_path, 'w') as f:
        f.write("# EEG to Text Translation Evaluation Report\n\n")
        
        # Summary
        f.write("## Evaluation Summary\n\n")
        summary = report['evaluation_summary']
        f.write(f"- **Evaluation Date**: {summary['timestamp']}\n")
        f.write(f"- **Models Evaluated**: {', '.join(summary['models_evaluated'])}\n")
        f.write(f"- **Total Test Samples**: {summary['total_test_samples']}\n\n")
        
        # Model Performance
        f.write("## Model Performance\n\n")
        for model_name, metrics in report['model_performance'].items():
            f.write(f"### {model_name}\n\n")
            for metric, value in metrics.items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            f.write("\n")
        
        # Model Comparison
        f.write("## Model Comparison\n\n")
        comparison = report['model_comparison']
        if 'best_models' in comparison:
            f.write("### Best Models by Metric\n\n")
            for metric, (model_name, score) in comparison['best_models'].items():
                f.write(f"- **{metric}**: {model_name} ({score:.4f})\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        for rec in report['recommendations']:
            f.write(f"- {rec}\n")
        f.write("\n")
        
        # Detailed Analysis
        f.write("## Detailed Analysis\n\n")
        analysis = report['detailed_analysis']
        
        if 'word_length' in analysis:
            f.write("### Performance by Word Length\n\n")
            f.write("Analysis of model performance based on target word length.\n\n")
        
        if 'error_patterns' in analysis:
            f.write("### Error Pattern Analysis\n\n")
            f.write("Analysis of common error patterns and failure cases.\n\n")
        
        if 'eeg_analysis' in analysis:
            f.write("### EEG Data Characteristics\n\n")
            f.write("Analysis of EEG data characteristics and their impact on performance.\n\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of EEG to text models')
    parser.add_argument('--lstm_bart_path', type=str, required=True,
                       help='Path to LSTM-BART model checkpoint')
    parser.add_argument('--bart_baseline_path', type=str, required=True,
                       help='Path to BART baseline model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Run comprehensive evaluation
    results = comprehensive_evaluation(
        lstm_bart_path=args.lstm_bart_path,
        bart_baseline_path=args.bart_baseline_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device
    )
    
    logger.info("Comprehensive evaluation completed!")


if __name__ == '__main__':
    main() 