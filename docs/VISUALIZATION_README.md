# SNN Optical Flow Visualization Tool

This script provides comprehensive visualization capabilities for testing trained Spiking Neural Network (SNN) optical flow models.

## Features

- **Model Inference**: Load trained models and run inference on dataset samples
- **Side-by-side Comparison**: Visualize ground truth vs. predicted optical flow
- **Error Analysis**: Show per-pixel flow error (EPE) heatmaps
- **Vector Field Plots**: Quiver plots showing flow direction and magnitude
- **Animation Generation**: Create GIF/MP4 animations showing predictions across multiple frames
- **Batch Processing**: Process multiple samples at once

## Usage

### Basic Visualization

Visualize predictions on specific samples:

```bash
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --config snn/configs/lightweight.yaml \
    --sample-idx 0 10 20 30 40 \
    --output-dir visualizations
```

### Create Animation

Generate an animated GIF showing predictions across a sequence:

```bash
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --config snn/configs/lightweight.yaml \
    --create-animation \
    --animation-samples 30 \
    --output-dir visualizations \
    --no-show
```

### Custom Dataset Path

Override the data root path from the config:

```bash
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --config snn/configs/lightweight.yaml \
    --data-root ../blink_sim/output/train_girl1 \
    --sample-idx 0 5 10 \
    --no-show
```

### Full Options

```bash
python visualize_predictions.py --help
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (default: `checkpoints/best_model.pth`)
- `--config`: Path to model config file (default: `snn/configs/lightweight.yaml`)
- `--data-root`: Override dataset path from config
- `--sample-idx`: List of sample indices to visualize (default: `[0, 10, 20, 30, 40]`)
- `--output-dir`: Directory to save visualizations (default: `visualizations`)
- `--create-animation`: Create animated GIF of predictions
- `--animation-samples`: Number of samples for animation (default: 20)
- `--no-show`: Don't display plots interactively (only save to disk)
- `--device`: Device for inference: `cuda` or `cpu` (default: auto-detect)

## Output Format

### Individual Sample Visualizations

Each visualization contains 6 subplots in a 2x3 grid:

**Top Row:**
1. Input Events (sum of event voxel grid)
2. Ground Truth Flow (color-coded by direction/magnitude)
3. Predicted Flow (color-coded by direction/magnitude)

**Bottom Row:**
1. Flow Error Heatmap (EPE per pixel)
2. Ground Truth Flow Vectors (quiver plot)
3. Predicted Flow Vectors (quiver plot)

**Metrics Displayed:**
- EPE (End-Point Error): Average pixel-wise flow error
- Max Flow GT: Maximum ground truth flow magnitude
- Max Flow Pred: Maximum predicted flow magnitude

### Animation Output

The animation shows three panels for each frame:
1. Input Events
2. Ground Truth Flow
3. Predicted Flow (with EPE displayed in title)

Each frame is labeled with the sequence name and frame index.

## Code Structure

The script is organized into clean, reusable functions:

### Core Functions

```python
load_config(config_path)
# Load YAML configuration file

load_model_from_checkpoint(checkpoint_path, config, device)
# Load trained model from checkpoint

load_dataset(config, data_root)
# Create dataset instance for inference

run_inference(model, sample, device)
# Run model inference and compute metrics

visualize_flow_comparison(input_events, flow_gt, flow_pred, metrics, ...)
# Create side-by-side visualization

create_flow_animation(model, dataset, indices, device, save_path, fps)
# Generate animated visualization
```

### Reused Components

The script leverages existing codebase utilities:
- `snn.models`: Model architectures (SpikingFlowNet, SpikingFlowNetLite)
- `snn.data`: Dataset loading (OpticalFlowDataset)
- `snn.utils.visualization`: Flow visualization utilities (flow_to_color, visualize_flow)

## Examples

### Quick Test on Best Model

```bash
# Visualize 5 samples from the best trained model
python visualize_predictions.py \
    --sample-idx 0 50 100 150 200 \
    --no-show
```

### Full Animation for Presentation

```bash
# Create a comprehensive 50-frame animation
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --create-animation \
    --animation-samples 50 \
    --no-show
```

### Compare Different Checkpoints

```bash
# Visualize with checkpoint from epoch 10
python visualize_predictions.py \
    --checkpoint checkpoints/checkpoint_epoch_10.pth \
    --sample-idx 0 25 50 75 \
    --output-dir visualizations/epoch10 \
    --no-show

# Visualize with best model
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --sample-idx 0 25 50 75 \
    --output-dir visualizations/best \
    --no-show
```

## Color Coding

The flow visualizations use the Middlebury color wheel convention:
- **Hue**: Flow direction (0° = right, 90° = down, 180° = left, 270° = up)
- **Brightness**: Flow magnitude (brighter = faster motion)

## Performance Notes

- Inference is fast (~10-20ms per sample on GPU)
- Visualization generation takes ~1-2s per sample
- Animation creation is memory-efficient (processes one frame at a time)
- For large datasets, use `--no-show` to avoid GUI overhead

## Tips

1. **Check Dataset Size**: The script shows how many samples are available
2. **Start Small**: Test with a few samples before creating large animations
3. **Use Best Model**: For publication-quality results, use `checkpoints/best_model.pth`
4. **Batch Processing**: Can specify many sample indices: `--sample-idx 0 10 20 30 40 50 60 70 80 90`
5. **Headless Mode**: Always use `--no-show` on remote servers without X11

## Integration with Training

After training completes, visualize results:

```bash
# Train the model
python train.py --config snn/configs/lightweight.yaml

# Visualize predictions from best model
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --config snn/configs/lightweight.yaml \
    --create-animation \
    --animation-samples 30
```

## Troubleshooting

**Issue**: "Sample X exceeds dataset size"
- **Solution**: Check dataset size in output, use valid indices

**Issue**: Model fails to load
- **Solution**: Ensure checkpoint and config match (same model type)

**Issue**: No GUI/display errors
- **Solution**: Use `--no-show` flag for headless operation

**Issue**: Out of memory
- **Solution**: Reduce `--animation-samples` or process fewer samples at once
