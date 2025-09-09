"""
Visualization Utilities for EEG to Text Translation

This module provides visualization functions for EEG data, training curves,
and model comparison results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def plot_eeg_sample(eeg_data: np.ndarray, 
                   title: str = "EEG Sample",
                   save_path: Optional[str] = None,
                   show: bool = True):
    """
    Plot a single EEG sample.
    
    Args:
        eeg_data: EEG data (time_points, channels)
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    time_points = np.arange(eeg_data.shape[0])
    
    # Plot each channel
    for i in range(min(10, eeg_data.shape[1])):  # Plot first 10 channels
        ax.plot(time_points, eeg_data[:, i], label=f'Channel {i+1}', alpha=0.7)
    
    ax.set_xlabel('Time Points')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(log_file: str, 
                        save_path: Optional[str] = None,
                        show: bool = True):
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training log file
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # This is a simplified version - in practice you'd parse the actual log file
    # or use TensorBoard logs
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Example training curves (replace with actual data)
    epochs = np.arange(1, 51)
    train_loss = 2.5 * np.exp(-epochs / 20) + 0.1 + 0.05 * np.random.randn(50)
    val_loss = 2.8 * np.exp(-epochs / 25) + 0.15 + 0.08 * np.random.randn(50)
    train_acc = 1 - 0.9 * np.exp(-epochs / 15) + 0.02 * np.random.randn(50)
    val_acc = 1 - 0.95 * np.exp(-epochs / 18) + 0.03 * np.random.randn(50)
    
    # Training loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    axes[0, 1].plot(epochs, val_loss, 'r-', label='Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training accuracy
    axes[1, 0].plot(epochs, train_acc, 'g-', label='Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1, 1].plot(epochs, val_acc, 'm-', label='Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(metrics_file: str,
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot model comparison from metrics file.
    
    Args:
        metrics_file: Path to metrics comparison file
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        comparison_data = json.load(f)
    
    # Extract data
    metrics = []
    models = []
    values = []
    
    for metric, model_scores in comparison_data.items():
        if metric == 'best_models':
            continue
        for model_name, score in model_scores.items():
            metrics.append(metric)
            models.append(model_name)
            values.append(score)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Metric': metrics,
        'Model': models,
        'Score': values
    })
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grouped bar plot
    pivot_df = df.pivot(index='Metric', columns='Model', values='Score')
    pivot_df.plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend(title='Models')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(predictions: List[str],
                         targets: List[str],
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot confusion matrix for word-level predictions.
    
    Args:
        predictions: List of predicted words
        targets: List of target words
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    
    # Encode words to integers
    le = LabelEncoder()
    all_words = list(set(predictions + targets))
    le.fit(all_words)
    
    pred_encoded = le.transform(predictions)
    target_encoded = le.transform(targets)
    
    # Compute confusion matrix
    cm = confusion_matrix(target_encoded, pred_encoded)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique words for labels
    unique_words = le.classes_
    if len(unique_words) > 20:  # Limit for readability
        unique_words = unique_words[:20]
        cm = cm[:20, :20]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_words, yticklabels=unique_words, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Word-Level Confusion Matrix')
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_weights(attention_weights: np.ndarray,
                          eeg_channels: List[str],
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plot attention weights for EEG channels.
    
    Args:
        attention_weights: Attention weights (num_heads, seq_len, seq_len)
        eeg_channels: List of EEG channel names
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot attention weights for first 4 heads
    num_heads = min(4, attention_weights.shape[0])
    
    for i in range(num_heads):
        ax = axes[i]
        
        # Average attention weights across sequence
        avg_attention = np.mean(attention_weights[i], axis=0)
        
        # Create bar plot
        channels = eeg_channels[:len(avg_attention)]
        bars = ax.bar(range(len(channels)), avg_attention)
        
        ax.set_xlabel('EEG Channels')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Attention Head {i+1}')
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Color bars by attention weight
        colors = plt.cm.Reds(avg_attention / np.max(avg_attention))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_examples(predictions: List[str],
                           targets: List[str],
                           num_examples: int = 10,
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    Plot prediction examples with targets.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        num_examples: Number of examples to show
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
    
    if num_examples == 1:
        axes = [axes]
    
    for i in range(min(num_examples, len(predictions))):
        ax = axes[i]
        
        # Create text comparison
        target_text = f"Target: {targets[i]}"
        pred_text = f"Prediction: {predictions[i]}"
        
        # Check if prediction is correct
        is_correct = targets[i].lower().strip() == predictions[i].lower().strip()
        color = 'green' if is_correct else 'red'
        
        ax.text(0.1, 0.7, target_text, fontsize=12, fontweight='bold')
        ax.text(0.1, 0.3, pred_text, fontsize=12, color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Example {i+1}', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_dashboard(results_dir: str, output_dir: str):
    """
    Create a comprehensive dashboard with all visualizations.
    
    Args:
        results_dir: Directory containing results
        output_dir: Directory to save dashboard
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find result files
    metrics_files = list(results_path.glob('*_metrics.json'))
    results_files = list(results_path.glob('*_results.json'))
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 15))
    
    # Title
    fig.suptitle('EEG to Text Translation Dashboard', fontsize=20, fontweight='bold')
    
    # Load and plot metrics
    if metrics_files:
        metrics_data = {}
        for file in metrics_files:
            with open(file, 'r') as f:
                data = json.load(f)
                model_name = data['model_name']
                metrics_data[model_name] = data['metrics']
        
        # Create metrics comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Model Metrics Comparison')
        ax1.set_ylabel('Score')
        plt.xticks(rotation=45)
    
    # Plot training curves (if available)
    ax2 = plt.subplot(2, 3, 2)
    # Add training curve plotting here
    
    # Plot prediction examples
    if results_files:
        with open(results_files[0], 'r') as f:
            results = json.load(f)
        
        ax3 = plt.subplot(2, 3, 3)
        predictions = results['predictions'][:5]
        targets = results['targets'][:5]
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            ax3.text(0.1, 0.8 - i*0.15, f"T: {target}", fontsize=10)
            ax3.text(0.1, 0.7 - i*0.15, f"P: {pred}", fontsize=10, 
                    color='green' if pred.lower() == target.lower() else 'red')
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Sample Predictions')
    
    plt.tight_layout()
    
    # Save dashboard
    plt.savefig(output_path / 'dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dashboard saved to {output_path / 'dashboard.png'}")


def main():
    """Main function for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations for EEG to text translation')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--type', type=str, choices=['dashboard', 'training', 'comparison'],
                       default='dashboard', help='Type of visualization')
    
    args = parser.parse_args()
    
    if args.type == 'dashboard':
        create_dashboard(args.results_dir, args.output_dir)
    elif args.type == 'training':
        # Add training curve plotting
        pass
    elif args.type == 'comparison':
        # Add model comparison plotting
        pass


if __name__ == '__main__':
    main() 