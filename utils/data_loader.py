"""
Data Loading Utilities for EEG to Text Translation

This module provides data loading functionality for the EEG to text translation task,
including PyTorch datasets and data loaders.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import BartTokenizer
import logging

logger = logging.getLogger(__name__)


class EEGTextDataset(Dataset):
    """PyTorch dataset for EEG to text translation."""
    
    def __init__(self, 
                 eeg_data_path: str,
                 words_path: str,
                 vocabulary_path: str,
                 tokenizer_name: str = 'facebook/bart-base',
                 max_length: int = 512,
                 pad_to_max: bool = True):
        """
        Initialize the EEG-Text dataset.
        
        Args:
            eeg_data_path: Path to EEG data .npy file
            words_path: Path to words .txt file
            vocabulary_path: Path to vocabulary .json file
            tokenizer_name: BART tokenizer name
            max_length: Maximum sequence length for text
            pad_to_max: Whether to pad EEG sequences to max length
        """
        self.eeg_data_path = Path(eeg_data_path)
        self.words_path = Path(words_path)
        self.vocabulary_path = Path(vocabulary_path)
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        
        # Load data
        self.eeg_data = np.load(self.eeg_data_path)
        
        # Load sequence lengths if available
        seq_lengths_path = self.eeg_data_path.parent / f"{self.eeg_data_path.stem.replace('_eeg', '_seq_lengths')}.npy"
        if seq_lengths_path.exists():
            self.seq_lengths = np.load(seq_lengths_path)
        else:
            self.seq_lengths = None
        
        # Load words
        with open(self.words_path, 'r') as f:
            self.words = [line.strip() for line in f.readlines()]
        
        # Load vocabulary
        with open(self.vocabulary_path, 'r') as f:
            vocab_data = json.load(f)
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']
            self.vocab_size = vocab_data['vocab_size']
        
        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens for EEG input
        special_tokens = ['<eeg_start>', '<eeg_end>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        logger.info(f"Dataset loaded: {len(self.eeg_data)} samples")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        logger.info(f"EEG data shape: {self.eeg_data.shape}")
    
    def __len__(self) -> int:
        return len(self.eeg_data)
    
    def pad_eeg_sequence(self, eeg_sequence: np.ndarray) -> np.ndarray:
        """
        Pad or truncate EEG sequence to max_length.
        
        Args:
            eeg_sequence: EEG data (sequence_length, num_channels)
            
        Returns:
            Padded/truncated EEG sequence
        """
        seq_len, num_channels = eeg_sequence.shape
        
        if seq_len > self.max_length:
            # Truncate
            return eeg_sequence[:self.max_length]
        elif seq_len < self.max_length and self.pad_to_max:
            # Pad with zeros
            padding = np.zeros((self.max_length - seq_len, num_channels))
            return np.vstack([eeg_sequence, padding])
        else:
            return eeg_sequence
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing EEG data and text tokens
        """
        # Get EEG data (already padded)
        eeg_sequence = self.eeg_data[idx]
        
        # Get sequence length for masking
        seq_length = self.seq_lengths[idx] if hasattr(self, 'seq_lengths') else eeg_sequence.shape[0]
        
        # Create attention mask for EEG data (1 for real data, 0 for padding)
        eeg_attention_mask = torch.zeros(eeg_sequence.shape[0], dtype=torch.long)
        eeg_attention_mask[:seq_length] = 1
        
        # Get target word
        target_word = self.words[idx]
        
        # Tokenize target word
        target_tokens = self.tokenizer(
            target_word,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create input text (for BART baseline - can be modified)
        input_text = f"<eeg_start> {target_word} <eeg_end>"
        input_tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'eeg_data': torch.FloatTensor(eeg_sequence),
            'eeg_attention_mask': eeg_attention_mask,
            'target_word': target_word,
            'target_input_ids': target_tokens['input_ids'].squeeze(0),
            'target_attention_mask': target_tokens['attention_mask'].squeeze(0),
            'input_ids': input_tokens['input_ids'].squeeze(0),
            'attention_mask': input_tokens['attention_mask'].squeeze(0),
            'word_idx': self.word_to_idx.get(target_word, 0),
            'seq_length': seq_length
        }


class EEGTextDataLoader:
    """Data loader wrapper for EEG to text translation."""
    
    def __init__(self, 
                 data_dir: str,
                 tokenizer_name: str = 'facebook/bart-base',
                 batch_size: int = 32,
                 max_length: int = 512,
                 num_workers: int = 4,
                 shuffle: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing processed data
            tokenizer_name: BART tokenizer name
            batch_size: Batch size for training
            max_length: Maximum sequence length
            num_workers: Number of worker processes
            shuffle: Whether to shuffle data
        """
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # Load vocabulary
        with open(self.data_dir / 'vocabulary.json', 'r') as f:
            vocab_data = json.load(f)
            self.vocab_size = vocab_data['vocab_size']
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']
        
        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        special_tokens = ['<eeg_start>', '<eeg_end>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        logger.info(f"DataLoader initialized with vocabulary size: {self.vocab_size}")
    
    def get_dataloader(self, split: str) -> DataLoader:
        """
        Get data loader for a specific split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            PyTorch DataLoader
        """
        splits_dir = self.data_dir / 'splits'
        
        eeg_data_path = splits_dir / f'{split}_eeg.npy'
        words_path = splits_dir / f'{split}_words.txt'
        vocabulary_path = self.data_dir / 'vocabulary.json'
        
        if not all([eeg_data_path.exists(), words_path.exists(), vocabulary_path.exists()]):
            raise FileNotFoundError(f"Missing data files for split: {split}")
        
        dataset = EEGTextDataset(
            eeg_data_path=str(eeg_data_path),
            words_path=str(words_path),
            vocabulary_path=str(vocabulary_path),
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if split == 'train' else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=split == 'train'
        )
        
        return dataloader
    
    def get_vocabulary_info(self) -> Dict:
        """Get vocabulary information."""
        return {
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }
    
    def get_tokenizer(self) -> BartTokenizer:
        """Get the BART tokenizer."""
        return self.tokenizer


class EEGFeatureDataset(Dataset):
    """Dataset for EEG features (alternative to raw EEG data)."""
    
    def __init__(self, 
                 features_path: str,
                 words_path: str,
                 vocabulary_path: str,
                 tokenizer_name: str = 'facebook/bart-base',
                 max_length: int = 512):
        """
        Initialize the EEG features dataset.
        
        Args:
            features_path: Path to EEG features .npy file
            words_path: Path to words .txt file
            vocabulary_path: Path to vocabulary .json file
            tokenizer_name: BART tokenizer name
            max_length: Maximum sequence length for text
        """
        self.features_path = Path(features_path)
        self.words_path = Path(words_path)
        self.vocabulary_path = Path(vocabulary_path)
        self.max_length = max_length
        
        # Load data
        self.features = np.load(self.features_path)
        
        # Load words
        with open(self.words_path, 'r') as f:
            self.words = [line.strip() for line in f.readlines()]
        
        # Load vocabulary
        with open(self.vocabulary_path, 'r') as f:
            vocab_data = json.load(f)
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']
            self.vocab_size = vocab_data['vocab_size']
        
        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        special_tokens = ['<eeg_start>', '<eeg_end>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        logger.info(f"Features dataset loaded: {len(self.features)} samples")
        logger.info(f"Features shape: {self.features.shape}")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing EEG features and text tokens
        """
        # Get EEG features
        features = self.features[idx]
        
        # Get target word
        target_word = self.words[idx]
        
        # Tokenize target word
        target_tokens = self.tokenizer(
            target_word,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'eeg_features': torch.FloatTensor(features),
            'target_word': target_word,
            'target_input_ids': target_tokens['input_ids'].squeeze(0),
            'target_attention_mask': target_tokens['attention_mask'].squeeze(0),
            'word_idx': self.word_to_idx.get(target_word, 0)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Separate different types of data
    eeg_data = [item['eeg_data'] for item in batch]
    target_input_ids = [item['target_input_ids'] for item in batch]
    target_attention_mask = [item['target_attention_mask'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    word_indices = [item['word_idx'] for item in batch]
    
    # Stack tensors
    batched_data = {
        'eeg_data': torch.stack(eeg_data),
        'target_input_ids': torch.stack(target_input_ids),
        'target_attention_mask': torch.stack(target_attention_mask),
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'word_indices': torch.LongTensor(word_indices)
    }
    
    return batched_data 