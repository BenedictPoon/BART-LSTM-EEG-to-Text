"""
EEG Data Preprocessing Utilities

This module handles the preprocessing of EEG data from the Zuco dataset,
including loading .npy files, extracting features, and preparing data for training.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Handles preprocessing of EEG data from Zuco dataset."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the EEG preprocessor.
        
        Args:
            data_dir: Directory containing the raw EEG .npy files
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # EEG parameters (adjust based on your dataset)
        self.sampling_rate = 1000  # Hz (typical for EEG)
        self.num_channels = 64     # Number of EEG channels
        self.max_sequence_length = 1000  # Maximum sequence length for padding
        
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse EEG filename to extract metadata.
        
        Args:
            filename: Filename in format 's{start}_e{end}_{word}.npy'
            
        Returns:
            Dictionary with start, end, and word information
        """
        try:
            # Remove .npy extension
            name = filename.replace('.npy', '')
            
            # Split by underscores
            parts = name.split('_')
            
            # Extract start and end samples
            start_sample = parts[0].replace('s', '')
            end_sample = parts[1].replace('e', '')
            
            # Extract word (join remaining parts in case word contains underscores)
            word = '_'.join(parts[2:])
            
            return {
                'start_sample': int(start_sample),
                'end_sample': int(end_sample),
                'word': word,
                'filename': filename
            }
        except Exception as e:
            logger.warning(f"Could not parse filename {filename}: {e}")
            return None
    
    def load_eeg_data(self, filepath: Path) -> Optional[np.ndarray]:
        """
        Load EEG data from .npy file.
        
        Args:
            filepath: Path to the .npy file
            
        Returns:
            EEG data as numpy array or None if loading fails
        """
        try:
            data = np.load(filepath)
            
            # Ensure data is 2D (samples x channels)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            
            return data
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def extract_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from EEG data.
        
        Args:
            eeg_data: Raw EEG data (samples x channels)
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Time-domain features
        features['mean'] = np.mean(eeg_data, axis=0)
        features['std'] = np.std(eeg_data, axis=0)
        features['min'] = np.min(eeg_data, axis=0)
        features['max'] = np.max(eeg_data, axis=0)
        features['range'] = features['max'] - features['min']
        
        # Frequency-domain features (simple FFT-based)
        fft_data = np.fft.fft(eeg_data, axis=0)
        power_spectrum = np.abs(fft_data) ** 2
        
        # Frequency bands (typical EEG bands)
        freqs = np.fft.fftfreq(eeg_data.shape[0], 1/self.sampling_rate)
        
        # Delta (0.5-4 Hz)
        delta_mask = (freqs >= 0.5) & (freqs <= 4)
        if np.any(delta_mask):
            features['delta_power'] = np.mean(power_spectrum[delta_mask], axis=0)
        else:
            features['delta_power'] = np.zeros(eeg_data.shape[1])
        
        # Theta (4-8 Hz)
        theta_mask = (freqs >= 4) & (freqs <= 8)
        if np.any(theta_mask):
            features['theta_power'] = np.mean(power_spectrum[theta_mask], axis=0)
        else:
            features['theta_power'] = np.zeros(eeg_data.shape[1])
        
        # Alpha (8-13 Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        if np.any(alpha_mask):
            features['alpha_power'] = np.mean(power_spectrum[alpha_mask], axis=0)
        else:
            features['alpha_power'] = np.zeros(eeg_data.shape[1])
        
        # Beta (13-30 Hz)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        if np.any(beta_mask):
            features['beta_power'] = np.mean(power_spectrum[beta_mask], axis=0)
        else:
            features['beta_power'] = np.zeros(eeg_data.shape[1])
        
        # Gamma (30-100 Hz)
        gamma_mask = (freqs >= 30) & (freqs <= 100)
        if np.any(gamma_mask):
            features['gamma_power'] = np.mean(power_spectrum[gamma_mask], axis=0)
        else:
            features['gamma_power'] = np.zeros(eeg_data.shape[1])
        
        return features
    
    def process_session(self, session_dir: Path) -> List[Dict]:
        """
        Process all EEG files in a session directory.
        
        Args:
            session_dir: Directory containing EEG files for one session
            
        Returns:
            List of processed data dictionaries
        """
        processed_data = []
        
        for filepath in session_dir.glob('*.npy'):
            # Parse filename
            metadata = self.parse_filename(filepath.name)
            if metadata is None:
                continue
            
            # Load EEG data
            eeg_data = self.load_eeg_data(filepath)
            if eeg_data is None:
                continue
            
            # Extract features
            features = self.extract_features(eeg_data)
            
            # Combine metadata and features
            sample_data = {
                **metadata,
                'session': session_dir.name,
                'eeg_data': eeg_data,
                'features': features,
                'sequence_length': eeg_data.shape[0]
            }
            
            processed_data.append(sample_data)
            
        return processed_data
    
    def normalize_data(self, data_list: List[Dict]) -> List[Dict]:
        """
        Normalize EEG data and features across all samples.
        
        Args:
            data_list: List of processed data dictionaries
            
        Returns:
            List of normalized data dictionaries
        """
        # Collect all EEG data for normalization
        all_eeg_data = np.vstack([item['eeg_data'] for item in data_list])
        
        # Fit scaler on all data
        scaler = StandardScaler()
        scaler.fit(all_eeg_data)
        
        # Normalize each sample
        for item in data_list:
            item['eeg_data_normalized'] = scaler.transform(item['eeg_data'])
            
            # Also normalize features
            feature_array = np.concatenate([
                item['features']['mean'],
                item['features']['std'],
                item['features']['delta_power'],
                item['features']['theta_power'],
                item['features']['alpha_power'],
                item['features']['beta_power'],
                item['features']['gamma_power']
            ])
            
            item['features_normalized'] = feature_array
        
        # Save scaler for later use
        import joblib
        joblib.dump(scaler, self.output_dir / 'eeg_scaler.pkl')
        
        return data_list
    
    def create_dataset_splits(self, data_list: List[Dict], 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """
        Create train/validation/test splits.
        
        Args:
            data_list: List of all processed data
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            
        Returns:
            Dictionary with train, val, and test splits
        """
        # Shuffle data
        np.random.shuffle(data_list)
        
        n_samples = len(data_list)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        splits = {
            'train': data_list[:train_end],
            'val': data_list[train_end:val_end],
            'test': data_list[val_end:]
        }
        
        logger.info(f"Dataset splits: Train={len(splits['train'])}, "
                   f"Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, List[Dict]]):
        """
        Save processed data to disk.
        
        Args:
            splits: Dictionary with train/val/test splits
        """
        splits_dir = self.output_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        for split_name, data_list in splits.items():
            # Save as JSON (metadata only) - convert numpy arrays to lists
            metadata_list = []
            for item in data_list:
                metadata_item = {}
                for k, v in item.items():
                    if k not in ['eeg_data', 'eeg_data_normalized', 'features']:
                        # Convert numpy types to Python types for JSON serialization
                        if isinstance(v, np.integer):
                            metadata_item[k] = int(v)
                        elif isinstance(v, np.floating):
                            metadata_item[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            metadata_item[k] = v.tolist()
                        else:
                            metadata_item[k] = v
                metadata_list.append(metadata_item)
            
            with open(splits_dir / f'{split_name}_metadata.json', 'w') as f:
                json.dump(metadata_list, f, indent=2)
            
            # Pad EEG data to uniform length and save as numpy arrays
            max_seq_len = max(item['eeg_data_normalized'].shape[0] for item in data_list)
            num_channels = data_list[0]['eeg_data_normalized'].shape[1]
            
            padded_eeg_data = np.zeros((len(data_list), max_seq_len, num_channels))
            for i, item in enumerate(data_list):
                seq_len = item['eeg_data_normalized'].shape[0]
                padded_eeg_data[i, :seq_len, :] = item['eeg_data_normalized']
            
            np.save(splits_dir / f'{split_name}_eeg.npy', padded_eeg_data)
            
            # Save features
            features = np.array([item['features_normalized'] for item in data_list])
            np.save(splits_dir / f'{split_name}_features.npy', features)
            
            # Save words
            words = [item['word'] for item in data_list]
            with open(splits_dir / f'{split_name}_words.txt', 'w') as f:
                for word in words:
                    f.write(word + '\n')
            
            # Save sequence lengths for later use
            seq_lengths = [item['sequence_length'] for item in data_list]
            np.save(splits_dir / f'{split_name}_seq_lengths.npy', np.array(seq_lengths))
            
            logger.info(f"{split_name}: {len(data_list)} samples, max_seq_len: {max_seq_len}, num_channels: {num_channels}")
        
        # Save vocabulary
        all_words = []
        for split_data in splits.values():
            all_words.extend([item['word'] for item in split_data])
        
        unique_words = sorted(list(set(all_words)))
        word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        vocab_data = {
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab_size': len(unique_words)
        }
        
        with open(self.output_dir / 'vocabulary.json', 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary size: {len(unique_words)}")
    
    def process_all_data(self):
        """Process all EEG data in the data directory."""
        logger.info("Starting EEG data preprocessing...")
        
        all_processed_data = []
        
        # Process each session directory
        for session_dir in self.data_dir.iterdir():
            if session_dir.is_dir():
                logger.info(f"Processing session: {session_dir.name}")
                session_data = self.process_session(session_dir)
                all_processed_data.extend(session_data)
        
        logger.info(f"Total samples processed: {len(all_processed_data)}")
        
        # Normalize data
        logger.info("Normalizing data...")
        all_processed_data = self.normalize_data(all_processed_data)
        
        # Create splits
        logger.info("Creating dataset splits...")
        splits = self.create_dataset_splits(all_processed_data)
        
        # Save processed data
        logger.info("Saving processed data...")
        self.save_processed_data(splits)
        
        logger.info("Preprocessing completed successfully!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Preprocess EEG data from Zuco dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing raw EEG .npy files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed data')
    parser.add_argument('--sampling_rate', type=int, default=1000,
                       help='EEG sampling rate in Hz')
    parser.add_argument('--num_channels', type=int, default=64,
                       help='Number of EEG channels')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(args.data_dir, args.output_dir)
    preprocessor.sampling_rate = args.sampling_rate
    preprocessor.num_channels = args.num_channels
    
    # Process data
    preprocessor.process_all_data()


if __name__ == '__main__':
    main() 