# Quantization-Aware Training (QAT) Guide for EventSNNFlowNetLiteV2

This guide explains how to quantize your fully-trained EventSNNFlowNetLiteV2 model to different bit-widths (8-bit, 4-bit, 2-bit, 1-bit) using quantization-aware training.

## Quick Start

### Step 1: Fine-tune with 8-bit Quantization

Start from your full-precision model in `checkpoints/V2/best_model.pth`:

```bash
python finetune_quantized.py \
    --config snn/configs/event_snn_lite_8bit.yaml \
    --pretrained checkpoints/V2/best_model.pth \
    --checkpoint-dir checkpoints/V2_8bit \
    --log-dir logs/V2_8bit
```

This will:
- Load your pre-trained full-precision model
- Enable 8-bit quantization layers
- Fine-tune for 30 epochs with reduced learning rate
- Save the best model to `checkpoints/V2_8bit/best_model.pth`

### Step 2: Fine-tune with 4-bit Quantization

Use the 8-bit model as starting point:

```bash
python finetune_quantized.py \
    --config snn/configs/event_snn_lite_4bit.yaml \
    --pretrained checkpoints/V2_8bit/best_model.pth \
    --checkpoint-dir checkpoints/V2_4bit \
    --log-dir logs/V2_4bit
```

### Step 3: Fine-tune with 2-bit Quantization

```bash
python finetune_quantized.py \
    --config snn/configs/event_snn_lite_2bit.yaml \
    --pretrained checkpoints/V2_4bit/best_model.pth \
    --checkpoint-dir checkpoints/V2_2bit \
    --log-dir logs/V2_2bit
```

### Step 4: (Optional) Binary 1-bit Quantization

```bash
python finetune_quantized.py \
    --config snn/configs/event_snn_lite_1bit.yaml \
    --pretrained checkpoints/V2_2bit/best_model.pth \
    --checkpoint-dir checkpoints/V2_1bit \
    --log-dir logs/V2_1bit
```

## Alternative: Progressive Quantization Schedule

Instead of fine-tuning separately for each bit-width, you can use a progressive schedule that automatically reduces bit-width during training:

```bash
python train.py \
    --config snn/configs/event_snn_lite_progressive.yaml \
    --resume checkpoints/V2/best_model.pth \
    --checkpoint-dir checkpoints/V2_progressive \
    --log-dir logs/V2_progressive
```

This will train for 90 epochs:
- Epochs 0-29: 8-bit quantization
- Epochs 30-59: 4-bit quantization  
- Epochs 60-89: 2-bit quantization

## Configuration Files

All quantization configs are in `snn/configs/`:

- `event_snn_lite_8bit.yaml` - 8-bit quantization (30 epochs, LR=5e-5)
- `event_snn_lite_4bit.yaml` - 4-bit quantization (30 epochs, LR=2e-5)
- `event_snn_lite_2bit.yaml` - 2-bit quantization (40 epochs, LR=1e-5)
- `event_snn_lite_1bit.yaml` - 1-bit/binary quantization (50 epochs, LR=5e-6)
- `event_snn_lite_progressive.yaml` - Progressive schedule (90 epochs)

### Key Configuration Parameters

Each config file includes:

```yaml
# Enable quantization
quantization_enabled: true
initial_bit_width: 8        # Target bit-width (8, 4, 2, or 1)
binarize: false             # Set true for 1-bit binary

# Fine-tuning hyperparameters (tuned for each bit-width)
learning_rate: 0.00005      # Lower than full-precision training
num_epochs: 30              # Fewer epochs for fine-tuning
quant_weight: 0.001         # Weight for quantization loss
```

## How It Works

### Quantization-Aware Training (QAT)

QAT simulates quantization during training by:

1. **Forward Pass**: Activations are quantized to the target bit-width
2. **Backward Pass**: Gradients flow through using straight-through estimators
3. **Weight Updates**: Weights remain in full precision during training
4. **Deployment**: Weights can be quantized to target bit-width for hardware

This allows the model to adapt to quantization constraints during training, resulting in much better accuracy than post-training quantization.

### Implementation Details

- **Quantization Layer**: `QuantizationAwareLayer` in `snn/quantization/quantization_aware.py`
- **Statistics Tracking**: Uses exponential moving average (EMA) to track min/max values
- **Symmetric Quantization**: Quantizes around zero for hardware efficiency
- **Straight-Through Estimators**: Allows gradient flow during backpropagation

### Model Architecture Support

Quantization is integrated into all spiking layers:
- `SpikingConvBlock` - Main building block
- `SpikingConv2d` - Convolutional layers
- `SpikingConvTranspose2d` - Upsampling layers

When `quantize=True`, each layer adds a `QuantizationAwareLayer` after convolution.

## Evaluation and Comparison

### Evaluate a Quantized Model

```bash
python tools/evaluate.py \
    --checkpoint checkpoints/V2_8bit/best_model.pth \
    --config snn/configs/event_snn_lite_8bit.yaml \
    --data-root ../blink_sim/output/test_set/
```

### Visualize Predictions

```bash
python tools/visualize_predictions.py \
    --checkpoint checkpoints/V2_8bit/best_model.pth \
    --config snn/configs/event_snn_lite_8bit.yaml \
    --data-root ../blink_sim/output/test_set/ \
    --sequence boy2_SweepFall_0
```

### Compare Different Bit-Widths

Create a comparison script:

```python
from snn.utils.quantization_utils import compare_model_outputs, estimate_model_size

# Load models with different bit-widths
model_fp = load_model('checkpoints/V2/best_model.pth', quantize=False)
model_8bit = load_model('checkpoints/V2_8bit/best_model.pth', bit_width=8)
model_4bit = load_model('checkpoints/V2_4bit/best_model.pth', bit_width=4)

# Compare outputs
metrics = compare_model_outputs(model_fp, model_8bit, test_input)
print(f"8-bit vs FP32 - MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.6f}")

# Estimate sizes
estimate_model_size(model_8bit)
estimate_model_size(model_4bit)
```

## Expected Results

### Model Size Reduction

For EventSNNFlowNetLiteV2 with base_ch=32 (~1.2M parameters):

| Bit-Width | Model Size | Size Reduction |
|-----------|------------|----------------|
| 32-bit (FP32) | ~4.8 MB | Baseline |
| 8-bit | ~1.2 MB | 4× smaller |
| 4-bit | ~0.6 MB | 8× smaller |
| 2-bit | ~0.3 MB | 16× smaller |
| 1-bit | ~0.15 MB | 32× smaller |

### Accuracy Trade-offs

Expected relative performance (approximate):

- **8-bit**: 95-99% of full-precision accuracy
- **4-bit**: 90-95% of full-precision accuracy
- **2-bit**: 80-90% of full-precision accuracy
- **1-bit**: 70-85% of full-precision accuracy

Actual results depend on your dataset and application requirements.

## Troubleshooting

### Issue: Model performance drops significantly

**Solution**: Try these approaches:
1. Use lower learning rate (reduce by 2-5×)
2. Train for more epochs
3. Start from higher bit-width checkpoint (e.g., 4-bit from 8-bit)
4. Increase `quant_weight` in config to penalize quantization more

### Issue: Training is unstable

**Solution**:
1. Reduce learning rate
2. Use tighter gradient clipping (e.g., `grad_clip: 0.5`)
3. Reduce batch size
4. Check that `quantization_enabled: true` in config

### Issue: "Missing keys" or "Unexpected keys" when loading checkpoint

**Solution**:
- This is normal when loading full-precision into quantized model
- Quantization layers (`quant_layer.*`) are initialized randomly
- Use `--strict-load` flag only if you're loading same architecture

### Issue: Want to skip 8-bit and go directly to 4-bit

**Not recommended**, but possible:
```bash
python finetune_quantized.py \
    --config snn/configs/event_snn_lite_4bit.yaml \
    --pretrained checkpoints/V2/best_model.pth \
    --checkpoint-dir checkpoints/V2_4bit_direct \
    --log-dir logs/V2_4bit_direct
```

Results will likely be worse than progressive quantization.

## Advanced Usage

### Custom Quantization Schedule

Edit the config file to create custom schedules:

```yaml
quantization_schedule:
  0: 8      # Start with 8-bit
  20: 6     # Switch to 6-bit at epoch 20
  40: 4     # Switch to 4-bit at epoch 40
  70: 2     # Switch to 2-bit at epoch 70
```

### Monitor Quantization Statistics

The trainer logs quantization statistics to TensorBoard:
- `train/quant_loss` - Quantization penalty
- `train/flow_loss` - Flow prediction loss
- Layer-specific quantization ranges

View with:
```bash
tensorboard --logdir logs/V2_8bit
```

### Export for Hardware Deployment

```python
from snn.utils.quantization_utils import export_quantized_model_info

model = load_model('checkpoints/V2_8bit/best_model.pth')
export_quantized_model_info(
    model,
    output_path='quantization_info.json',
    include_weights=False
)
```

## References

- **QAT Implementation**: `snn/quantization/quantization_aware.py`
- **Fine-tuning Script**: `finetune_quantized.py`
- **Trainer with QAT**: `snn/training/trainer.py` (see `update_quantization()`)
- **Utilities**: `snn/utils/quantization_utils.py`

## Next Steps

1. **Start with 8-bit**: This gives best accuracy with 4× size reduction
2. **Evaluate on your test set**: Check if accuracy is acceptable
3. **Try 4-bit**: If 8-bit is good, push further to 4-bit for 8× reduction
4. **Hardware deployment**: Export quantized model for FPGA/ASIC deployment

Good luck with your quantization experiments!
