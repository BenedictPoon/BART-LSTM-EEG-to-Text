# Quick Start Guide - EEG to Text Translation Pipeline

## 🚀 **Step 1: Install Dependencies**

```bash
# Option 1: Use the installation script
./install_dependencies.sh

# Option 2: Install manually
pip install torch torchvision transformers accelerate
pip install numpy scipy scikit-learn pyyaml tqdm
pip install nltk rouge-score matplotlib seaborn
python -c "import nltk; nltk.download('punkt')"
```

## 🧠 **Step 2: Verify Data Processing**

Your data has already been processed successfully! Here's what was created:

```
✅ 2,688 EEG samples processed
✅ Dataset splits: Train=1,881, Val=403, Test=404
✅ EEG data: 105 channels, max 500 time points
✅ Vocabulary: 493 unique words
```

## 🧪 **Step 3: Test the Pipeline**

```bash
# Test data loading
python test_preprocessing.py
```

## 🎯 **Step 4: Start Training**

### **Option A: Train BART Baseline (Recommended to start)**
```bash
python training/train_bart_baseline.py --config configs/bart_config.yaml
```

### **Option B: Train LSTM-BART**
```bash
python training/train_lstm_bart.py --config configs/lstm_bart_config.yaml
```

## 📊 **Step 5: Monitor Training**

Training logs will be saved to:
- `logs/training_baseline.log` (for BART baseline)
- `logs/training.log` (for LSTM-BART)

## 🔍 **Step 6: Evaluate Models**

After training completes:
```bash
# Run comprehensive evaluation
python inference/evaluation.py \
    --lstm_bart_path models/saved/lstm_bart_best.pt \
    --bart_baseline_path models/saved/bart_baseline_best.pt \
    --data_dir data/processed/ \
    --output_dir evaluation_results/
```

## ⚙️ **Configuration Files**

- **BART Baseline**: `configs/bart_config.yaml`
- **LSTM-BART**: `configs/lstm_bart_config.yaml`

Key settings:
- `eeg_input_size: 105` (your EEG channels)
- `batch_size: 32` (adjust based on memory)
- `num_epochs: 50` (training duration)

## 🐛 **Troubleshooting**

### **Common Issues:**

1. **Missing dependencies**: Run `./install_dependencies.sh`
2. **Memory issues**: Reduce `batch_size` in config files
3. **CUDA not available**: Models will automatically use CPU
4. **Log directory error**: Already fixed - directories created automatically

## 📈 **Expected Results**

With your dataset (2,688 samples, 105 channels):
- **BART Baseline**: Good baseline performance
- **LSTM-BART**: Should show improved temporal modeling
- **Metrics**: BLEU, ROUGE, accuracy scores

## 🎉 **Success Indicators**

✅ Preprocessing completed (2,688 samples)  
✅ Data loading works  
✅ Training starts without errors  
✅ Models save checkpoints  
✅ Evaluation generates reports  

---

**Ready to start?** Run the training command above! 🚀

