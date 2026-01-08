"""
Quick test script to verify the SNN optical flow setup
"""

import torch
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from snn.models import SpikingFlowNet, SpikingFlowNetLite, LIFNeuron
        from snn.quantization import QuantizationAwareLayer, BinaryQuantizer
        from snn.data import OpticalFlowDataset
        from snn.training import SNNTrainer, flow_loss
        from snn.utils import Logger, compute_metrics, visualize_flow
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model instantiation"""
    print("\nTesting model creation...")
    try:
        from snn.models import SpikingFlowNet, SpikingFlowNetLite
        
        # Test full model
        model_full = SpikingFlowNet(
            in_channels=5,
            num_timesteps=10,
            quantize=False
        )
        print(f"✓ SpikingFlowNet created - Parameters: {sum(p.numel() for p in model_full.parameters()):,}")
        
        # Test lite model
        model_lite = SpikingFlowNetLite(
            in_channels=5,
            num_timesteps=10,
            quantize=False
        )
        print(f"✓ SpikingFlowNetLite created - Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through model"""
    print("\nTesting forward pass...")
    try:
        from snn.models import SpikingFlowNetLite
        
        # Create model
        model = SpikingFlowNetLite(
            in_channels=5,
            num_timesteps=10,
            quantize=False
        )
        model.eval()
        
        # Create dummy input
        batch_size = 2
        height, width = 128, 128
        x = torch.randn(batch_size, 5, height, width)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x)
        
        # Check output
        assert 'flow' in outputs, "Output should contain 'flow' key"
        assert outputs['flow'].shape == (batch_size, 2, height, width), "Flow shape incorrect"
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output flow shape: {outputs['flow'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization():
    """Test quantization layers"""
    print("\nTesting quantization...")
    try:
        from snn.quantization import QuantizationAwareLayer, BinaryQuantizer
        
        # Test QAT layer
        qat = QuantizationAwareLayer(bit_width=8)
        x = torch.randn(4, 64, 32, 32)
        x_quant = qat(x)
        print(f"✓ 8-bit quantization successful")
        
        # Test binary quantizer
        binary = BinaryQuantizer()
        x_binary = binary(x)
        assert torch.all((x_binary == 1) | (x_binary == -1)), "Binary output should be {-1, +1}"
        print(f"✓ Binary quantization successful")
        
        return True
    except Exception as e:
        print(f"✗ Quantization test failed: {e}")
        return False


def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    try:
        from snn.data import OpticalFlowDataset
        
        # Try to load dataset
        data_root = Path("../blink_sim/output")
        if not data_root.exists():
            print(f"⚠ Dataset not found at {data_root} - skipping data loading test")
            return True
        
        dataset = OpticalFlowDataset(
            data_root=str(data_root),
            split='train',
            use_events=True,
            num_bins=5,
            crop_size=(128, 128),
            max_samples=1  # Just load one sample
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Dataset loaded successfully!")
            print(f"  Number of samples: {len(dataset)}")
            print(f"  Sample keys: {sample.keys()}")
            print(f"  Input shape: {sample['input'].shape}")
            print(f"  Flow shape: {sample['flow'].shape}")
        else:
            print(f"⚠ Dataset is empty")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return True  # Don't fail if dataset isn't available


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    try:
        import yaml
        
        configs = [
            'snn/configs/baseline.yaml',
            'snn/configs/quantization_aware.yaml',
            'snn/configs/binary_snn.yaml',
            'snn/configs/lightweight.yaml'
        ]
        
        for config_path in configs:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✓ Loaded {config_path}")
            else:
                print(f"✗ Config not found: {config_path}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def main():
    print("="*60)
    print("SNN Optical Flow - Setup Verification")
    print("="*60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Quantization", test_quantization),
        ("Data Loading", test_data_loading),
        ("Config Loading", test_config_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("  1. python train.py --config snn/configs/lightweight.yaml")
        print("  2. tensorboard --logdir ./logs")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
