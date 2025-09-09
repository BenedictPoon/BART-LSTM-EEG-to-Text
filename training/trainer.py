"""
Training Utilities for EEG to Text Translation

This module provides training functionality for both LSTM-BART and BART baseline models,
including training loops, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import time
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb

from utils.metrics import compute_metrics
from utils.data_loader import EEGTextDataLoader

logger = logging.getLogger(__name__)


class EEGTrainer:
    """Trainer class for EEG to text translation models."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 tokenizer,
                 device: str = 'cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            tokenizer: BART tokenizer
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.device = device
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.gradient_clip_val = config['training']['gradient_clip_val']
        self.accumulation_steps = config['training']['accumulation_steps']
        
        # Validation parameters
        self.eval_every_n_epochs = config['validation']['eval_every_n_epochs']
        self.metrics = config['validation']['metrics']
        
        # Logging parameters
        self.log_every_n_steps = config['logging']['log_every_n_steps']
        self.use_tensorboard = config['logging']['tensorboard']
        self.use_wandb = config['logging']['wandb']
        
        # Paths
        self.model_save_dir = Path(config['paths']['model_save_dir'])
        self.logs_dir = Path(config['paths']['logs_dir'])
        self.results_dir = Path(config['paths']['results_dir'])
        
        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.early_stopping_counter = 0
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        self.early_stopping_min_delta = config['training']['early_stopping']['min_delta']
        
        # Move model to device
        self.model.to(device)
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = self.config['training']['warmup_steps']
        scheduler_type = self.config['training']['scheduler']['type']
        
        if scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer and scheduler initialized: {scheduler_type}")
    
    def _setup_logging(self):
        """Setup logging (TensorBoard, WandB)."""
        # TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.logs_dir / 'tensorboard')
        
        # Weights & Biases
        if self.use_wandb:
            wandb.init(
                project=self.config['logging']['project_name'],
                config=self.config
            )
        
        logger.info("Logging setup completed")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # Progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                eeg_data=batch['eeg_data'],
                target_ids=batch['target_input_ids'],
                attention_mask=batch['target_attention_mask']
            )
            
            loss = outputs['loss']
            
            # Backward pass with gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_val
                )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                self._log_training_step(loss.item())
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    eeg_data=batch['eeg_data'],
                    target_ids=batch['target_input_ids'],
                    attention_mask=batch['target_attention_mask']
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Generate predictions for metrics
                generated_ids = self.model.generate(
                    eeg_data=batch['eeg_data'],
                    max_length=self.config['model']['max_length']
                )
                
                # Decode predictions and targets
                predictions = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                targets = self.tokenizer.batch_decode(
                    batch['target_input_ids'], 
                    skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        metrics = compute_metrics(all_predictions, all_targets, self.metrics)
        metrics['val_loss'] = avg_loss
        
        return metrics
    
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        if self.use_tensorboard:
            self.tb_writer.add_scalar('train/loss', loss, self.global_step)
            self.tb_writer.add_scalar('train/learning_rate', 
                                    self.optimizer.param_groups[0]['lr'], 
                                    self.global_step)
        
        if self.use_wandb:
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'global_step': self.global_step
            })
    
    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch metrics."""
        if self.use_tensorboard:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'train/{key}', value, self.current_epoch)
            
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        if self.use_wandb:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'epoch': self.current_epoch
            })
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.model_save_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = self.model_save_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            logger.info(f"Best model saved: {best_model_path}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        max_checkpoints = self.config['training']['max_checkpoints']
        checkpoints = sorted(self.model_save_dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > max_checkpoints:
            for checkpoint in checkpoints[:-max_checkpoints]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_metrics = self.validate()
                
                # Check if this is the best model
                is_best = val_metrics['val_loss'] < self.best_val_loss - self.early_stopping_min_delta
                
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_val_metrics = val_metrics.copy()
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
            
            # Logging
            self._log_epoch(train_metrics, val_metrics)
            
            # Log to console
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
                for metric, value in val_metrics.items():
                    if metric != 'val_loss':
                        logger.info(f"Val {metric}: {value:.4f}")
            
            # Save checkpoint
            save_this_epoch = (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0
            if save_this_epoch or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint()
        
        # Close logging
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return self.best_val_metrics


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: Dict) -> str:
    """Get device based on configuration."""
    device_config = config['hardware']['device']
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    else:
        return device_config 