#!/usr/bin/env python3
"""
Inference Script for EEG to Text Translation

This script loads trained models and generates text from EEG data.
"""

import argparse
import logging
import sys
import torch
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_bart import LSTMBartModel
from models.bart_baseline import BartBaselineModel
from utils.data_loader import EEGTextDataLoader
from utils.metrics import compute_metrics, print_metrics_summary, save_metrics_to_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, model_type: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('lstm_bart' or 'bart_baseline')
        device: Device to load model on
        
    Returns:
        Loaded model
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
    
    logger.info(f"Model loaded successfully. Epoch: {checkpoint['epoch']}")
    return model


def generate_text(model: torch.nn.Module, 
                 eeg_data: torch.Tensor,
                 tokenizer,
                 max_length: int = 512,
                 num_beams: int = 4,
                 temperature: float = 1.0,
                 do_sample: bool = False) -> List[str]:
    """
    Generate text from EEG data.
    
    Args:
        model: Trained model
        eeg_data: EEG data tensor
        tokenizer: BART tokenizer
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        List of generated texts
    """
    model.eval()
    
    with torch.no_grad():
        # Generate token IDs
        generated_ids = model.generate(
            eeg_data=eeg_data,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample
        )
        
        # Decode to text
        generated_texts = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
    
    return generated_texts


def evaluate_model(model: torch.nn.Module,
                  test_dataloader,
                  tokenizer,
                  model_name: str,
                  output_dir: str,
                  max_length: int = 512) -> Dict[str, float]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_dataloader: Test data loader
        tokenizer: BART tokenizer
        model_name: Name of the model
        output_dir: Directory to save results
        max_length: Maximum generation length
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on test data...")
    
    all_predictions = []
    all_targets = []
    all_eeg_data = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate predictions
            predictions = generate_text(
                model=model,
                eeg_data=batch['eeg_data'],
                tokenizer=tokenizer,
                max_length=max_length
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
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    save_metrics_to_file(
        metrics=metrics,
        filepath=str(output_path / f'{model_name}_metrics.json'),
        model_name=model_name
    )
    
    # Save predictions
    results = {
        'predictions': all_predictions,
        'targets': all_targets,
        'metrics': metrics
    }
    
    with open(output_path / f'{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save some examples
    examples = []
    for i in range(min(20, len(all_predictions))):
        examples.append({
            'index': i,
            'target': all_targets[i],
            'prediction': all_predictions[i],
            'eeg_shape': all_eeg_data[i].shape
        })
    
    with open(output_path / f'{model_name}_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return metrics


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained EEG to text models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['lstm_bart', 'bart_baseline'],
                       help='Type of model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--do_sample', action='store_true',
                       help='Use sampling instead of beam search')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.model_type, device)
    
    # Create data loader
    data_loader = EEGTextDataLoader(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_dataloader = data_loader.get_dataloader('test')
    tokenizer = data_loader.get_tokenizer()
    
    logger.info(f"Test samples: {len(test_dataloader.dataset)}")
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
        model_name=args.model_type,
        output_dir=args.output_dir,
        max_length=args.max_length
    )
    
    logger.info("Inference completed!")


if __name__ == '__main__':
    main() 