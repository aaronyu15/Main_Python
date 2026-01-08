# Main Python - SNN Optical Flow

## Directory Structure

```
Main_Python/
├── snn/                    # Core SNN package
│   ├── configs/           # Configuration files
│   ├── data/              # Dataset loaders
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
│
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
│
├── debug/                 # Debugging utilities
│   ├── debug_dataset.py  # Dataset visualization
│   ├── debug_flow.py     # Model prediction analysis
│   └── debug_outputs/    # Debug visualizations
│
├── tools/                 # Utility scripts
│   ├── visualize_predictions.py  # Inference & visualization
│   ├── evaluate.py              # Model evaluation
│   └── test_setup.py            # Setup verification
│
├── output/                # Training outputs
│   ├── visualizations/   # Prediction visualizations
│   └── results/          # Evaluation results
│
├── docs/                  # Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── SETUP_COMPLETE.md
│   └── VISUALIZATION_README.md
│
├── checkpoints/           # Current model checkpoints
├── logs/                  # Training logs
└── archives/              # Old checkpoints and logs
    ├── checkpoints_old/
    └── logs_old/
```

## Quick Commands

### Training
```bash
python train.py --config snn/configs/lightweight.yaml --data-root ../blink_sim/output/train_girl1
```

### Debugging
```bash
# Visualize dataset samples
python debug/debug_dataset.py

# Analyze model predictions
python debug/debug_flow.py
```

### Visualization
```bash
# Create prediction visualizations
python tools/visualize_predictions.py --checkpoint checkpoints/best_model.pth --config snn/configs/lightweight.yaml --create-animation --sequence girl1_BaseballHit_0
```

### Evaluation
```bash
python tools/evaluate.py --checkpoint checkpoints/best_model.pth --config snn/configs/lightweight.yaml
```

## Output Locations

- **Training checkpoints**: `checkpoints/`
- **Training logs**: `logs/`
- **Visualizations**: `output/visualizations/`
- **Debug outputs**: `debug/debug_outputs/`
- **Old runs**: `archives/`
