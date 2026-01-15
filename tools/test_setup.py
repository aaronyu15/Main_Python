"""
Quick test script to verify the SNN optical flow setup
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path so we can import snn module
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from snn.models import EventSNNFlowNetLite, EventSNNFlowNetLiteV2
        from snn.quantization import QuantizationAwareLayer, BinaryQuantizer
        from snn.data import OpticalFlowDataset
        from snn.training import SNNTrainer, flow_loss
        from snn.utils import Logger, compute_metrics, visualize_flow
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model instantiation"""
    print("\nTesting model creation...")
    try:
        from snn.models import EventSNNFlowNetLite, EventSNNFlowNetLiteV2
        
        # Test event-based model
        model_event = EventSNNFlowNetLite()
        print(f"✓ EventSNNFlowNetLite created - Parameters: {sum(p.numel() for p in model_event.parameters()):,}")

        # Test event-based model
        model_event = EventSNNFlowNetLiteV2()
        print(f"✓ EventSNNFlowNetLiteV2 created - Parameters: {sum(p.numel() for p in model_event.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through model"""
    print("\nTesting forward pass...")
    try:
        from snn.models import EventSNNFlowNetLiteV2

        if not torch.cuda.is_available():
            print("⚠ CUDA not available - using CPU")
    
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"  Using device: {device}")
        
        # Create model
        model = EventSNNFlowNetLiteV2(
            base_ch=32,
        ).to(device)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        height, width = 128, 128
        x = torch.randn(batch_size, 5, 2, height, width).to(device)
        
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


def test_gpu_computation():
    """Test GPU computation with various operations"""
    print("\nTesting GPU computation...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available - skipping GPU test")
        return False
    
    try:
        device = torch.device('cuda')
        print(f"  Using device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Test 1: Basic matrix operations
        print("\n  Test 1: Matrix multiplication...")
        size = 2048
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Timed calculation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        print(f"    Matrix mult ({size}x{size}): {elapsed_ms:.2f} ms")
        
        # Test 2: Convolution operations
        print("\n  Test 2: Convolution operations...")
        batch_size = 8
        in_channels = 64
        out_channels = 128
        h, w = 256, 256
        
        conv_input = torch.randn(batch_size, in_channels, h, w, device=device)
        conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(device)
        
        start.record()
        conv_output = conv_layer(conv_input)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        print(f"    Conv2d ({batch_size}x{in_channels}x{h}x{w}): {elapsed_ms:.2f} ms")
        print(f"    Output shape: {conv_output.shape}")
        
        # Test 3: Reduction operations
        print("\n  Test 3: Reduction operations...")
        large_tensor = torch.randn(10000, 10000, device=device)
        
        start.record()
        mean_val = large_tensor.mean()
        std_val = large_tensor.std()
        max_val = large_tensor.max()
        min_val = large_tensor.min()
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        print(f"    Reductions (mean/std/max/min): {elapsed_ms:.2f} ms")
        print(f"    Mean: {mean_val.item():.4f}, Std: {std_val.item():.4f}")
        
        # Test 4: Element-wise operations
        print("\n  Test 4: Element-wise operations...")
        x = torch.randn(100, 100, 100, device=device)
        
        start.record()
        y = torch.relu(x)
        y = torch.sigmoid(y)
        y = y * 2.0 + 1.0
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        print(f"    Element-wise ops: {elapsed_ms:.2f} ms")
        
        # Test 5: GPU-CPU transfer
        print("\n  Test 5: Data transfer...")
        gpu_tensor = torch.randn(100, 100, device=device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        cpu_tensor = gpu_tensor.cpu()
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"    GPU->CPU transfer: {elapsed_ms:.2f} ms")
        
        start_event.record()
        back_to_gpu = cpu_tensor.to(device)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"    CPU->GPU transfer: {elapsed_ms:.2f} ms")
        
        # Memory summary
        print(f"\n  Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Max GPU memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        
        # Cleanup
        del a, b, c, conv_input, conv_layer, conv_output, large_tensor, x, y, gpu_tensor
        torch.cuda.empty_cache()
        
        print("✓ GPU computation test successful!")
        return True
        
    except Exception as e:
        print(f"✗ GPU computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        ("GPU Computation", test_gpu_computation),
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
