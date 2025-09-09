#!/usr/bin/env python3
"""
Training Script for BART Baseline EEG to Text Translation

This script trains the BART baseline model for EEG to text translation.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.bart_baseline import create_bart_baseline_model
from utils.data_loader import EEGTextDataLoader
from training.trainer import EEGTrainer, load_config, set_random_seed, get_device

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_baseline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BART baseline model for EEG to text translation')
    parser.add_argument('--config', type=str, default='configs/bart_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory from config')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['paths']['model_save_dir'] = args.output_dir
    
    # Set random seed
    set_random_seed(config['random_seed'])
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Create data loader
    logger.info("Initializing data loader...")
    data_loader = EEGTextDataLoader(
        data_dir=config['data']['data_dir'],
        tokenizer_name=config['data']['tokenizer_name'],
        batch_size=config['data']['batch_size'],
        max_length=config['data']['max_length'],
        num_workers=config['data']['num_workers'],
        shuffle=config['data']['shuffle']
    )
    
    # Get train and validation dataloaders
    train_dataloader = data_loader.get_dataloader('train')
    val_dataloader = data_loader.get_dataloader('val')
    
    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Create model
    logger.info("Creating BART baseline model...")
    model = create_bart_baseline_model(config['model'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = EEGTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=data_loader.get_tokenizer(),
        device=device
    )
    
    # Resume training if checkpoint provided
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    best_metrics = trainer.train()
    
    # Print final results
    logger.info("Training completed!")
    logger.info("Best validation metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save final configuration
    config_path = Path(config['paths']['model_save_dir']) / 'final_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Final configuration saved to {config_path}")


if __name__ == '__main__':
    main() 