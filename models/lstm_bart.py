"""
LSTM-BART Hybrid Model for EEG to Text Translation

This module implements a hybrid model that uses an LSTM encoder to process EEG signals
and a BART decoder to generate text output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartConfig, BartForConditionalGeneration
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """LSTM encoder for processing EEG time series data."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        """
        Initialize LSTM encoder.
        
        Args:
            input_size: Number of EEG channels
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(
            hidden_size * self.num_directions,
            hidden_size
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"LSTM Encoder initialized: input_size={input_size}, "
                   f"hidden_size={hidden_size}, num_layers={num_layers}, "
                   f"bidirectional={bidirectional}")
    
    def forward(self, eeg_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM encoder.
        
        Args:
            eeg_data: EEG data tensor (batch_size, seq_len, input_size)
            
        Returns:
            Tuple of (encoded_features, hidden_states)
        """
        batch_size, seq_len, _ = eeg_data.shape
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        
        if eeg_data.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(eeg_data, (h0, c0))
        
        # Use the last hidden state from all layers and directions
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            last_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Project to output size
        encoded_features = self.output_projection(last_hidden)
        encoded_features = self.layer_norm(encoded_features)
        
        return encoded_features, lstm_out


class BridgeLayer(nn.Module):
    """Bridge layer to connect LSTM output to BART input."""
    
    def __init__(self, 
                 lstm_hidden_size: int,
                 bart_hidden_size: int = 768,
                 dropout: float = 0.1):
        """
        Initialize bridge layer.
        
        Args:
            lstm_hidden_size: Size of LSTM output
            bart_hidden_size: Size of BART hidden states
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.bart_hidden_size = bart_hidden_size
        
        # Linear projection from LSTM to BART hidden size
        self.projection = nn.Linear(lstm_hidden_size, bart_hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(bart_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional: Add positional encoding for BART
        self.add_positional_encoding = True
        
        logger.info(f"Bridge Layer initialized: {lstm_hidden_size} -> {bart_hidden_size}")
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bridge layer.
        
        Args:
            lstm_output: LSTM encoded features (batch_size, lstm_hidden_size)
            
        Returns:
            BART-compatible hidden states (batch_size, 1, bart_hidden_size)
        """
        # Project to BART hidden size
        projected = self.projection(lstm_output)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        # Add sequence dimension for BART (batch_size, 1, bart_hidden_size)
        bart_hidden = projected.unsqueeze(1)
        
        return bart_hidden


class LSTMBartModel(nn.Module):
    """Hybrid LSTM-BART model for EEG to text translation."""
    
    def __init__(self, 
                 eeg_input_size: int,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 2,
                 bart_model_name: str = 'facebook/bart-base',
                 max_length: int = 512,
                 dropout: float = 0.1,
                 use_features: bool = False,
                 feature_size: int = 0):
        """
        Initialize LSTM-BART model.
        
        Args:
            eeg_input_size: Number of EEG channels
            lstm_hidden_size: LSTM hidden size
            lstm_num_layers: Number of LSTM layers
            bart_model_name: Pre-trained BART model name
            max_length: Maximum sequence length
            dropout: Dropout rate
            use_features: Whether to use additional EEG features
            feature_size: Size of additional features
        """
        super().__init__()
        
        self.eeg_input_size = eeg_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.max_length = max_length
        self.use_features = use_features
        
        # LSTM encoder
        self.lstm_encoder = LSTMEncoder(
            input_size=eeg_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Feature processing (if using additional features)
        if use_features and feature_size > 0:
            self.feature_encoder = nn.Linear(feature_size, lstm_hidden_size)
            self.feature_layer_norm = nn.LayerNorm(lstm_hidden_size)
            self.feature_dropout = nn.Dropout(dropout)
        
        # Bridge layer
        self.bridge = BridgeLayer(
            lstm_hidden_size=lstm_hidden_size,
            bart_hidden_size=768,  # BART base hidden size
            dropout=dropout
        )
        
        # BART decoder
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Freeze BART encoder (we only use the decoder)
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"LSTM-BART Model initialized with {bart_model_name}")
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize LSTM weights
        for name, param in self.lstm_encoder.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize bridge layer weights
        if self.bridge.projection.weight.dim() >= 2:
            nn.init.xavier_uniform_(self.bridge.projection.weight)
        nn.init.zeros_(self.bridge.projection.bias)
        
        # Initialize feature encoder weights if it exists
        if self.use_features and hasattr(self, 'feature_encoder'):
            if self.feature_encoder.weight.dim() >= 2:
                nn.init.xavier_uniform_(self.feature_encoder.weight)
            nn.init.zeros_(self.feature_encoder.bias)
        
        logger.info("Model weights initialized")
    
    def forward(self, 
                eeg_data: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None,
                eeg_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LSTM-BART model.
        
        Args:
            eeg_data: EEG data (batch_size, seq_len, eeg_input_size)
            target_ids: Target text token IDs (batch_size, target_len)
            eeg_features: Additional EEG features (batch_size, feature_size)
            attention_mask: Attention mask for target sequence
            
        Returns:
            Dictionary containing loss and logits
        """
        batch_size = eeg_data.shape[0]
        
        # LSTM encoding
        lstm_output, _ = self.lstm_encoder(eeg_data)
        
        # Process additional features if provided
        if self.use_features and eeg_features is not None:
            feature_encoded = self.feature_encoder(eeg_features)
            feature_encoded = self.feature_layer_norm(feature_encoded)
            feature_encoded = self.feature_dropout(feature_encoded)
            
            # Combine LSTM output with features
            lstm_output = lstm_output + feature_encoded
        
        # Bridge to BART
        bart_hidden = self.bridge(lstm_output)
        
        # Create encoder attention mask (all ones for single token)
        encoder_attention_mask = torch.ones(batch_size, 1, dtype=torch.long, device=eeg_data.device)
        
        if target_ids is not None:
            # Training mode: compute loss
            outputs = self.bart(
                inputs_embeds=bart_hidden,
                attention_mask=encoder_attention_mask,
                labels=target_ids,
                decoder_attention_mask=attention_mask
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'lstm_output': lstm_output,
                'bart_hidden': bart_hidden
            }
        else:
            # Inference mode: generate text
            outputs = self.bart(
                inputs_embeds=bart_hidden,
                attention_mask=encoder_attention_mask,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.bart.config.pad_token_id,
                eos_token_id=self.bart.config.eos_token_id
            )
            
            return {
                'generated_ids': outputs.sequences,
                'lstm_output': lstm_output,
                'bart_hidden': bart_hidden
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
        
        # LSTM encoding
        lstm_output, _ = self.lstm_encoder(eeg_data)
        
        # Process features if provided
        if self.use_features and eeg_features is not None:
            feature_encoded = self.feature_encoder(eeg_features)
            feature_encoded = self.feature_layer_norm(feature_encoded)
            feature_encoded = self.feature_dropout(feature_encoded)
            lstm_output = lstm_output + feature_encoded
        
        # Bridge to BART
        bart_hidden = self.bridge(lstm_output)
        
        # Create attention mask
        batch_size = eeg_data.shape[0]
        encoder_attention_mask = torch.ones(batch_size, 1, dtype=torch.long, device=eeg_data.device)
        
        # Generate text
        generated_ids = self.bart.generate(
            inputs_embeds=bart_hidden,
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
        Get LSTM encoder output for analysis.
        
        Args:
            eeg_data: EEG data
            
        Returns:
            LSTM encoded features
        """
        with torch.no_grad():
            lstm_output, _ = self.lstm_encoder(eeg_data)
            return lstm_output


def create_lstm_bart_model(config: Dict) -> LSTMBartModel:
    """
    Create LSTM-BART model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized LSTM-BART model
    """
    model = LSTMBartModel(
        eeg_input_size=config.get('eeg_input_size', 64),
        lstm_hidden_size=config.get('lstm_hidden_size', 512),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        bart_model_name=config.get('bart_model_name', 'facebook/bart-base'),
        max_length=config.get('max_length', 512),
        dropout=config.get('dropout', 0.1),
        use_features=config.get('use_features', False),
        feature_size=config.get('feature_size', 0)
    )
    
    return model 