"""
BART Baseline Model for EEG to Text Translation

This module implements a baseline BART model that directly processes EEG data
without the LSTM encoder for comparison with the LSTM-BART hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartConfig, BartForConditionalGeneration
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EEGEncoder(nn.Module):
    """Simple encoder for EEG data to BART input."""
    
    def __init__(self, 
                 eeg_input_size: int,
                 bart_hidden_size: int = 768,
                 max_sequence_length: int = 512,
                 dropout: float = 0.1):
        """
        Initialize EEG encoder.
        
        Args:
            eeg_input_size: Number of EEG channels
            bart_hidden_size: BART hidden size
            max_sequence_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.eeg_input_size = eeg_input_size
        self.bart_hidden_size = bart_hidden_size
        self.max_sequence_length = max_sequence_length
        
        # Linear projection from EEG to BART hidden size
        self.eeg_projection = nn.Linear(eeg_input_size, bart_hidden_size)
        
        # Positional encoding for sequence
        self.positional_encoding = nn.Embedding(max_sequence_length, bart_hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(bart_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"EEG Encoder initialized: {eeg_input_size} -> {bart_hidden_size}")
    
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEG encoder.
        
        Args:
            eeg_data: EEG data (batch_size, seq_len, eeg_input_size)
            
        Returns:
            BART-compatible hidden states (batch_size, seq_len, bart_hidden_size)
        """
        batch_size, seq_len, _ = eeg_data.shape
        
        # Project EEG data to BART hidden size
        projected = self.eeg_projection(eeg_data)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=eeg_data.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoding = self.positional_encoding(positions)
        
        # Combine projection and positional encoding
        encoded = projected + pos_encoding
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        return encoded


class BartBaselineModel(nn.Module):
    """BART baseline model for EEG to text translation."""
    
    def __init__(self, 
                 eeg_input_size: int,
                 bart_model_name: str = 'facebook/bart-base',
                 max_length: int = 512,
                 dropout: float = 0.1,
                 use_features: bool = False,
                 feature_size: int = 0):
        """
        Initialize BART baseline model.
        
        Args:
            eeg_input_size: Number of EEG channels
            bart_model_name: Pre-trained BART model name
            max_length: Maximum sequence length
            dropout: Dropout rate
            use_features: Whether to use additional EEG features
            feature_size: Size of additional features
        """
        super().__init__()
        
        self.eeg_input_size = eeg_input_size
        self.max_length = max_length
        self.use_features = use_features
        
        # EEG encoder
        self.eeg_encoder = EEGEncoder(
            eeg_input_size=eeg_input_size,
            bart_hidden_size=768,  # BART base hidden size
            max_sequence_length=max_length,
            dropout=dropout
        )
        
        # Feature processing (if using additional features)
        if use_features and feature_size > 0:
            self.feature_encoder = nn.Linear(feature_size, 768)
            self.feature_layer_norm = nn.LayerNorm(768)
            self.feature_dropout = nn.Dropout(dropout)
        
        # BART model
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Freeze BART encoder (we replace it with our EEG encoder)
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"BART Baseline Model initialized with {bart_model_name}")
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize EEG encoder weights
        nn.init.xavier_uniform_(self.eeg_encoder.eeg_projection.weight)
        nn.init.zeros_(self.eeg_encoder.eeg_projection.bias)
        
        # Initialize feature encoder if used
        if self.use_features:
            nn.init.xavier_uniform_(self.feature_encoder.weight)
            nn.init.zeros_(self.feature_encoder.bias)
        
        logger.info("Model weights initialized")
    
    def forward(self, 
                eeg_data: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None,
                eeg_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BART baseline model.
        
        Args:
            eeg_data: EEG data (batch_size, seq_len, eeg_input_size)
            target_ids: Target text token IDs (batch_size, target_len)
            eeg_features: Additional EEG features (batch_size, feature_size)
            attention_mask: Attention mask for target sequence
            
        Returns:
            Dictionary containing loss and logits
        """
        batch_size, seq_len, _ = eeg_data.shape
        
        # EEG encoding
        eeg_encoded = self.eeg_encoder(eeg_data)
        
        # Process additional features if provided
        if self.use_features and eeg_features is not None:
            feature_encoded = self.feature_encoder(eeg_features)
            feature_encoded = self.feature_layer_norm(feature_encoded)
            feature_encoded = self.feature_dropout(feature_encoded)
            
            # Expand features to match sequence length
            feature_encoded = feature_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine EEG encoding with features
            eeg_encoded = eeg_encoded + feature_encoded
        
        # Create encoder attention mask
        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=eeg_data.device)
        
        if target_ids is not None:
            # Training mode: compute loss
            outputs = self.bart(
                inputs_embeds=eeg_encoded,
                attention_mask=encoder_attention_mask,
                labels=target_ids,
                decoder_attention_mask=attention_mask
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'eeg_encoded': eeg_encoded
            }
        else:
            # Inference mode: generate text
            outputs = self.bart(
                inputs_embeds=eeg_encoded,
                attention_mask=encoder_attention_mask,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.bart.config.pad_token_id,
                eos_token_id=self.bart.config.eos_token_id
            )
            
            return {
                'generated_ids': outputs.sequences,
                'eeg_encoded': eeg_encoded
            }
    
    def generate(self, 
                eeg_data: torch.Tensor,
                eeg_features: Optional[torch.Tensor] = None,
                max_length: Optional[int] = None,
                num_beams: int = 4,
                temperature: float = 1.0,
                do_sample: bool = False) -> torch.Tensor:
        """
        Generate text from EEG data.
        
        Args:
            eeg_data: EEG data (batch_size, seq_len, eeg_input_size)
            eeg_features: Additional EEG features
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated token IDs
        """
        if max_length is None:
            max_length = self.max_length
        
        # EEG encoding
        eeg_encoded = self.eeg_encoder(eeg_data)
        
        # Process features if provided
        if self.use_features and eeg_features is not None:
            feature_encoded = self.feature_encoder(eeg_features)
            feature_encoded = self.feature_layer_norm(feature_encoded)
            feature_encoded = self.feature_dropout(feature_encoded)
            
            # Expand features to match sequence length
            seq_len = eeg_data.shape[1]
            feature_encoded = feature_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            eeg_encoded = eeg_encoded + feature_encoded
        
        # Create attention mask
        batch_size, seq_len, _ = eeg_encoded.shape
        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=eeg_data.device)
        
        # Generate text
        generated_ids = self.bart.generate(
            inputs_embeds=eeg_encoded,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.bart.config.pad_token_id,
            eos_token_id=self.bart.config.eos_token_id,
            early_stopping=True
        )
        
        return generated_ids
    
    def get_encoder_output(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Get EEG encoder output for analysis.
        
        Args:
            eeg_data: EEG data
            
        Returns:
            EEG encoded features
        """
        with torch.no_grad():
            return self.eeg_encoder(eeg_data)


def create_bart_baseline_model(config: Dict) -> BartBaselineModel:
    """
    Create BART baseline model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized BART baseline model
    """
    model = BartBaselineModel(
        eeg_input_size=config.get('eeg_input_size', 64),
        bart_model_name=config.get('bart_model_name', 'facebook/bart-base'),
        max_length=config.get('max_length', 512),
        dropout=config.get('dropout', 0.1),
        use_features=config.get('use_features', False),
        feature_size=config.get('feature_size', 0)
    )
    
    return model 