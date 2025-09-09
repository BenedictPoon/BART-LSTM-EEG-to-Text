# LSTM-BART EEG to Text Translation Pipeline

This project implements a hybrid LSTM-BART model for translating EEG signals to text, with comparison against a baseline BART-only model.

## Project Overview

The pipeline consists of:
1. **LSTM-BART Model**: LSTM encoder for EEG feature extraction + BART decoder for text generation
2. **BART Baseline**: Direct BART model for EEG to text translation
3. **Data Processing**: EEG signal preprocessing and text tokenization
4. **Training Pipeline**: Complete training workflow with validation
5. **Inference Pipeline**: Model evaluation and text generation

## Dataset Structure

The Zuco 1 NR task dataset contains EEG recordings for individual words:
- **Format**: `s{start_sample}_e{end_sample}_{word}.npy`
- **Content**: EEG time series data for each word
- **Subjects**: ZAB (multiple recording sessions: NR1, NR3, NR5, NR6, NR2, NR4)

## Project Structure

```
lstm-bart/
├── data/
│   ├── raw/                    # Original .npy files
│   ├── processed/              # Preprocessed EEG data
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── lstm_bart.py           # LSTM-BART hybrid model
│   ├── bart_baseline.py       # BART-only baseline
│   └── config.py              # Model configurations
├── utils/
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # EEG preprocessing
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Plotting utilities
├── training/
│   ├── train_lstm_bart.py     # LSTM-BART training script
│   ├── train_bart_baseline.py # BART baseline training
│   └── trainer.py             # Training utilities
├── inference/
│   ├── inference.py           # Model inference script
│   └── evaluation.py          # Model evaluation
├── configs/
│   ├── lstm_bart_config.yaml  # LSTM-BART configuration
│   └── bart_config.yaml       # BART configuration
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv eeg_translation_env
source eeg_translation_env/bin/activate  # On macOS/Linux
# or
eeg_translation_env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Extract and organize data
python utils/preprocessing.py --data_dir ZAB/ --output_dir data/processed/
```

### 3. Training
```bash
# Train LSTM-BART model
python training/train_lstm_bart.py --config configs/lstm_bart_config.yaml

# Train BART baseline
python training/train_bart_baseline.py --config configs/bart_config.yaml
```

### 4. Inference
```bash
# Run inference on test data
python inference/inference.py --model_path models/lstm_bart_best.pt --data_path data/splits/test/
```

## Model Architecture

### LSTM-BART Hybrid
- **LSTM Encoder**: Processes EEG time series → fixed-size representation
- **BART Decoder**: Generates text from LSTM output
- **Bridge Layer**: Connects LSTM output to BART input

### BART Baseline
- **Direct Encoding**: EEG → BART input (with preprocessing)
- **BART Decoder**: Standard BART text generation

## Key Features

- **Modular Design**: Easy to experiment with different architectures
- **Configurable**: YAML-based configuration for hyperparameters
- **Reproducible**: Fixed random seeds and logging
- **Scalable**: Ready for larger datasets and distributed training
- **Evaluation**: Multiple metrics (BLEU, ROUGE, METEOR, etc.)

## Performance Comparison

The pipeline will compare:
1. **LSTM-BART**: Hybrid approach leveraging temporal EEG patterns
2. **BART Baseline**: Direct translation without temporal modeling

## Notes for Ubuntu Deployment

When moving to Ubuntu:
1. Update CUDA paths in configs
2. Adjust batch sizes for GPU memory
3. Use distributed training for larger datasets
4. Consider mixed precision training for efficiency

## Citation

If you use this code, please cite:
```
@misc{eeg_translation_pipeline,
  title={LSTM-BART EEG to Text Translation Pipeline},
  author={Your Name},
  year={2024}
}
``` 