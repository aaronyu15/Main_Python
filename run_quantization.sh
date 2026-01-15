#!/bin/bash
# Quick Start Script for Quantization-Aware Fine-tuning
# This script demonstrates the recommended workflow for quantizing your EventSNNFlowNetLiteV2 model

set -e  # Exit on error

echo "=============================================================================="
echo "Quantization-Aware Fine-tuning - Quick Start"
echo "=============================================================================="
echo ""
echo "This script will fine-tune your full-precision model (checkpoints/V2/best_model.pth)"
echo "to create quantized versions at different bit-widths."
echo ""
echo "Workflow:"
echo "  1. Full-precision (32-bit) → 8-bit quantization"
echo "  2. 8-bit → 4-bit quantization"  
echo "  3. (Optional) 4-bit → 2-bit quantization"
echo "  4. (Optional) 2-bit → 1-bit (binary) quantization"
echo ""
echo "=============================================================================="

# Check if pretrained model exists
PRETRAINED="checkpoints/V2/best_model.pth"
if [ ! -f "$PRETRAINED" ]; then
    echo "Error: Pretrained model not found at $PRETRAINED"
    echo "Please ensure you have a trained full-precision model first."
    exit 1
fi

echo "Found pretrained model: $PRETRAINED"
echo ""

# Ask user which bit-widths to train
read -p "Train 8-bit model? (y/n) " -n 1 -r TRAIN_8BIT
echo ""
read -p "Train 4-bit model? (y/n) " -n 1 -r TRAIN_4BIT
echo ""
read -p "Train 2-bit model? (y/n) " -n 1 -r TRAIN_2BIT
echo ""
read -p "Train 1-bit (binary) model? (y/n) " -n 1 -r TRAIN_1BIT
echo ""

# 8-bit quantization
if [[ $TRAIN_8BIT =~ ^[Yy]$ ]]; then
    echo ""
    echo "=============================================================================="
    echo "Step 1: Fine-tuning with 8-bit quantization"
    echo "=============================================================================="
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --pretrained checkpoints/V2/best_model.pth \
        --checkpoint-dir checkpoints/V2_8bit \
        --log-dir logs/V2_8bit
    
    echo ""
    echo "✓ 8-bit model saved to: checkpoints/V2_8bit/best_model.pth"
    PREV_MODEL="checkpoints/V2_8bit/best_model.pth"
else
    echo "Skipping 8-bit training"
    PREV_MODEL="checkpoints/V2/best_model.pth"
fi

# 4-bit quantization
if [[ $TRAIN_4BIT =~ ^[Yy]$ ]]; then
    echo ""
    echo "=============================================================================="
    echo "Step 2: Fine-tuning with 4-bit quantization"
    echo "=============================================================================="
    
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Warning: Previous model not found. Using full-precision model."
        PREV_MODEL="checkpoints/V2/best_model.pth"
    fi
    
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_4bit.yaml \
        --pretrained $PREV_MODEL \
        --checkpoint-dir checkpoints/V2_4bit \
        --log-dir logs/V2_4bit
    
    echo ""
    echo "✓ 4-bit model saved to: checkpoints/V2_4bit/best_model.pth"
    PREV_MODEL="checkpoints/V2_4bit/best_model.pth"
else
    echo "Skipping 4-bit training"
fi

# 2-bit quantization
if [[ $TRAIN_2BIT =~ ^[Yy]$ ]]; then
    echo ""
    echo "=============================================================================="
    echo "Step 3: Fine-tuning with 2-bit quantization"
    echo "=============================================================================="
    
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Warning: Previous model not found. Using full-precision model."
        PREV_MODEL="checkpoints/V2/best_model.pth"
    fi
    
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_2bit.yaml \
        --pretrained $PREV_MODEL \
        --checkpoint-dir checkpoints/V2_2bit \
        --log-dir logs/V2_2bit
    
    echo ""
    echo "✓ 2-bit model saved to: checkpoints/V2_2bit/best_model.pth"
    PREV_MODEL="checkpoints/V2_2bit/best_model.pth"
else
    echo "Skipping 2-bit training"
fi

# 1-bit (binary) quantization
if [[ $TRAIN_1BIT =~ ^[Yy]$ ]]; then
    echo ""
    echo "=============================================================================="
    echo "Step 4: Fine-tuning with 1-bit (binary) quantization"
    echo "=============================================================================="
    
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Warning: Previous model not found. Using full-precision model."
        PREV_MODEL="checkpoints/V2/best_model.pth"
    fi
    
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_1bit.yaml \
        --pretrained $PREV_MODEL \
        --checkpoint-dir checkpoints/V2_1bit \
        --log-dir logs/V2_1bit
    
    echo ""
    echo "✓ 1-bit model saved to: checkpoints/V2_1bit/best_model.pth"
else
    echo "Skipping 1-bit training"
fi

echo ""
echo "=============================================================================="
echo "Quantization-Aware Fine-tuning Complete!"
echo "=============================================================================="
echo ""
echo "Trained models:"
if [[ $TRAIN_8BIT =~ ^[Yy]$ ]]; then
    echo "  - 8-bit: checkpoints/V2_8bit/best_model.pth"
fi
if [[ $TRAIN_4BIT =~ ^[Yy]$ ]]; then
    echo "  - 4-bit: checkpoints/V2_4bit/best_model.pth"
fi
if [[ $TRAIN_2BIT =~ ^[Yy]$ ]]; then
    echo "  - 2-bit: checkpoints/V2_2bit/best_model.pth"
fi
if [[ $TRAIN_1BIT =~ ^[Yy]$ ]]; then
    echo "  - 1-bit: checkpoints/V2_1bit/best_model.pth"
fi
echo ""
echo "Next steps:"
echo "  1. Evaluate models on test set:"
echo "     python tools/evaluate.py --checkpoint checkpoints/V2_8bit/best_model.pth --config snn/configs/event_snn_lite_8bit.yaml"
echo ""
echo "  2. Visualize predictions:"
echo "     python tools/visualize_predictions.py --checkpoint checkpoints/V2_8bit/best_model.pth --config snn/configs/event_snn_lite_8bit.yaml"
echo ""
echo "  3. View training logs:"
echo "     tensorboard --logdir logs/"
echo ""
echo "See QUANTIZATION_GUIDE.md for detailed documentation."
echo "=============================================================================="
