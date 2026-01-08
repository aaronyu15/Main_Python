# Spiking Neural Network for Optical Flow Estimation

**Thesis Project: FPGA-Ready Spiking Neural Network for Event-Based Optical Flow**

This repository contains a complete implementation of a Spiking Neural Network (SNN) for optical flow estimation, designed for eventual deployment on FPGAs. The framework supports quantization-aware training with progressive bit-width reduction, enabling efficient hardware implementation.

## 🎯 Project Overview

- **Goal**: Train a spiking neural network for optical flow estimation that can be deployed on FPGAs
- **Key Features**:
  - Event-based optical flow estimation using SNNs
  - Quantization-aware training (32-bit → 8-bit → 4-bit → 1-bit)
  - Binary SNN support for extreme efficiency
  - Modular architecture for easy experimentation
  - Hardware-aware design with sparsity constraints

## 📁 Project Structure

```
Main_Python/
├── snn/                          # Main package
│   ├── models/                   # Neural network models
│   │   ├── snn_layers.py        # LIF neurons, spiking convolutions
│   │   └── spiking_flownet.py   # FlowNet architecture for SNNs
│   ├── quantization/            # Quantization utilities
│   │   ├── quantization_aware.py # QAT layers and methods
│   │   └── binary_layers.py     # Binary/XNOR layers for FPGA
│   ├── data/                    # Data loading and preprocessing
│   │   ├── optical_flow_dataset.py
│   │   └── data_utils.py
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           # Main training loop
│   │   └── losses.py            # Loss functions
│   ├── utils/                   # Utilities
│   │   ├── logger.py            # Logging and tensorboard
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── visualization.py     # Flow visualization
│   └── configs/                 # Configuration files
│       ├── baseline.yaml        # Standard training
│       ├── quantization_aware.yaml  # Progressive quantization
│       ├── binary_snn.yaml      # Binary network
│       └── lightweight.yaml     # Fast prototyping
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── checkpoints/                 # Saved models
├── logs/                        # Training logs
└── results/                     # Evaluation results
```

## 🚀 Getting Started

### 1. Installation

```bash
# Navigate to the project
cd Main_Python

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

The project uses datasets from the `blink_sim` directory. The expected structure is already set up at:
`../blink_sim/output/train/`

### 3. Training

#### Quick Start (Baseline)
```bash
python train.py --config snn/configs/baseline.yaml
```

#### Quantization-Aware Training
```bash
python train.py --config snn/configs/quantization_aware.yaml
```

#### Binary SNN Training (for FPGA)
```bash
python train.py --config snn/configs/binary_snn.yaml
```

#### Fast Prototyping (Lightweight)
```bash
python train.py --config snn/configs/lightweight.yaml
```

### 4. Evaluation

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --split train \
  --output-dir ./results \
  --save-visualizations
```

### 5. Monitoring Training

```bash
# Launch tensorboard
tensorboard --logdir ./logs
```

## 🔧 Configuration Guide

### Model Types

1. **SpikingFlowNet**: Full-featured model with encoder-decoder architecture
2. **SpikingFlowNetLite**: Lightweight model for FPGA deployment

### Quantization Strategies

#### Progressive Quantization (Recommended)
- Epochs 0-49: 32-bit (full precision)
- Epochs 50-99: 8-bit quantization
- Epochs 100-149: 4-bit quantization  
- Epochs 150+: 2-bit or binary

#### Binary SNN (Extreme Efficiency)
- 1-bit weights and activations
- XNOR-based operations for FPGA
- Minimal power consumption

## 📊 Key Features

### Quantization-Aware Training
All models support switchable quantization:
- Set `quantization_enabled: true` in config
- Define quantization schedule by epoch
- Progressive bit-width reduction (32→8→4→1)

### Hardware-Ready Design
- Sparsity constraints for power efficiency
- Binary layers for XNOR operations
- Configurable spike rates
- FPGA-friendly architectures

### Comprehensive Metrics
- Endpoint Error (EPE)
- Outlier percentage
- Angular error
- Spike activity statistics

## 🎓 Training Tips

### For Best Accuracy
1. Start with baseline configuration
2. Train for 200+ epochs
3. Monitor spike rate (should be 5-15%)

### For FPGA Deployment
1. Use `SpikingFlowNetLite` model
2. Enable quantization-aware training
3. Target low spike rates (<10%)
4. Use binary configuration

## 📈 Expected Performance

| Configuration | Model Size | Spike Rate | FPGA Suitability |
|--------------|------------|------------|------------------|
| Baseline (FP32) | 200 MB | 10% | Low |
| 8-bit QAT | 50 MB | 8% | Medium |
| 4-bit QAT | 25 MB | 7% | High |
| Binary SNN | 6 MB | 5% | Excellent |

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Use `SpikingFlowNetLite`
- Reduce `num_timesteps`

### Poor Convergence
- Increase learning rate
- Disable quantization initially
- Reduce `sparsity_weight`

## ✅ Quick Start Checklist

1. ✓ Project structure created
2. ✓ Install dependencies: `pip install -r requirements.txt`
3. ✓ Verify dataset access at `../blink_sim/output/train/`
4. ✓ Run test training: `python train.py --config snn/configs/lightweight.yaml`
5. ✓ Monitor with tensorboard: `tensorboard --logdir ./logs`

## 📝 Citation

If you use this code for your research, please cite your thesis.

Good luck with your thesis project! 🎓