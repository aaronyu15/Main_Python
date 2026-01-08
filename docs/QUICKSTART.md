# SNN Optical Flow - Quick Start Guide

## Installation (First Time Setup)

### Option 1: Automated Setup (Linux/Mac)
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

## Quick Test Run

Once setup is complete, test the system:

```bash
# Activate environment (if not already active)
source venv/bin/activate

# Quick test training (uses limited data for speed)
python train.py --config snn/configs/lightweight.yaml

# Monitor in another terminal
tensorboard --logdir ./logs
```

## Training Workflows

### 1. Baseline Training (Recommended First)
```bash
# Train full precision model
python train.py \
  --config snn/configs/baseline.yaml \
  --checkpoint-dir ./checkpoints/baseline \
  --log-dir ./logs/baseline

# This will train for 200 epochs
# Expected time: Several hours to days depending on GPU
```

### 2. Quantization-Aware Training
```bash
# Train with progressive quantization
python train.py \
  --config snn/configs/quantization_aware.yaml \
  --checkpoint-dir ./checkpoints/qat \
  --log-dir ./logs/qat

# Quantization schedule:
#   Epochs 0-49: 32-bit
#   Epochs 50-99: 8-bit
#   Epochs 100-149: 4-bit
#   Epochs 150+: 2-bit
```

### 3. Binary SNN (For FPGA)
```bash
# Train binary network
python train.py \
  --config snn/configs/binary_snn.yaml \
  --checkpoint-dir ./checkpoints/binary \
  --log-dir ./logs/binary

# This creates a 1-bit network ready for FPGA deployment
```

### 4. Fast Prototyping
```bash
# Quick experiments with limited data
python train.py \
  --config snn/configs/lightweight.yaml \
  --checkpoint-dir ./checkpoints/test \
  --log-dir ./logs/test

# Uses only 1000 training samples
# Good for testing changes before full training
```

## Evaluation

### Evaluate Trained Model
```bash
# Evaluate on validation/test set
python evaluate.py \
  --checkpoint ./checkpoints/baseline/best_model.pth \
  --split train \
  --output-dir ./results/baseline \
  --save-visualizations

# Results will be saved in ./results/baseline/
# Including flow visualizations and metrics
```

## Monitoring Training

### TensorBoard
```bash
# Launch tensorboard
tensorboard --logdir ./logs

# Open browser to: http://localhost:6006
```

### Check Training Progress
```bash
# View latest log
tail -f logs/*/training.log

# Check checkpoints
ls -lh checkpoints/
```

## Common Tasks

### Resume Training
```bash
# Resume from checkpoint
python train.py \
  --config snn/configs/baseline.yaml \
  --resume ./checkpoints/baseline/checkpoint_epoch_50.pth
```

### Export Model for FPGA
```bash
# After training binary model
python -c "
import torch
checkpoint = torch.load('checkpoints/binary/best_model.pth')
torch.save(checkpoint['model_state_dict'], 'model_for_fpga.pth')
print('Model exported to model_for_fpga.pth')
"
```

### Visualize Results
```bash
# Create flow visualizations from evaluation
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --split train \
  --output-dir ./results \
  --save-visualizations \
  --num-samples 10
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config file
# Or use CPU (slower):
python train.py --config snn/configs/lightweight.yaml --device cpu
```

### Dataset Not Found
```bash
# Verify dataset path
ls ../blink_sim/output/train/

# Or specify custom path
python train.py \
  --config snn/configs/baseline.yaml \
  --data-root /path/to/your/dataset
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Development Workflow

### 1. Make Changes to Code
```bash
# Edit files in snn/ directory
# e.g., snn/models/spiking_flownet.py
```

### 2. Quick Test
```bash
# Run test script
python test_setup.py

# Or quick training test
python train.py --config snn/configs/lightweight.yaml
```

### 3. Full Training
```bash
# Once satisfied, run full training
python train.py --config snn/configs/baseline.yaml
```

## File Structure Reference

```
Main_Python/
├── snn/                    # Main package
│   ├── models/            # Neural network architectures
│   ├── quantization/      # Quantization utilities
│   ├── data/              # Data loading
│   ├── training/          # Training loops and losses
│   ├── utils/             # Helper functions
│   └── configs/           # YAML configurations
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
├── test_setup.py          # Setup verification
├── setup.sh               # Automated setup
└── requirements.txt       # Dependencies
```

## Next Steps

1. **Verify Setup**: Run `python test_setup.py`
2. **Quick Test**: Run lightweight config
3. **Baseline Training**: Train full model
4. **Experiment**: Try different quantization levels
5. **Deploy**: Export binary model for FPGA

## Getting Help

- Check README.md for detailed documentation
- Run `python train.py --help` for command options
- Run `python evaluate.py --help` for evaluation options

Good luck with your thesis! 🎓
