#!/usr/bin/env python3
"""
Quick test script to validate all MLP models can be instantiated and run forward passes.
This helps debug issues before running full training.
"""

import torch
import torch.nn as nn
from models.mlp import MLP
from models.mlp_snn import MLPSNN
from models.branchy_mlp import BranchyMLP
from models.quantized_snn_mlp import QuantizedSNNMLP
from models.branchy_quantized_snn_mlp import BranchyQuantizedSNNMLP

def test_model(model_class, model_name, input_tensor, **kwargs):
    """Test a model by running a forward pass."""
    print(f"\n--- Testing {model_name} ---")
    try:
        model = model_class(**kwargs)
        model.eval()  # Set to eval mode to avoid BatchNorm issues
        
        print(f"Model created successfully")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                if len(output) == 3:
                    exit_outputs, final_output, exit_points = output
                    print(f"Branchy output - Exit outputs: {len(exit_outputs)}, Final: {final_output.shape}")
                elif len(output) == 2:
                    exit_outputs, final_output = output
                    print(f"Branchy output - Exit outputs: {len(exit_outputs)}, Final: {final_output.shape}")
                else:
                    print(f"Output shape: {output[0].shape if hasattr(output[0], 'shape') else type(output[0])}")
            else:
                if len(output.shape) == 3:  # SNN output [time, batch, classes]
                    print(f"SNN output shape: {output.shape} (time, batch, classes)")
                    output_summed = output.sum(dim=0)
                    print(f"Summed output shape: {output_summed.shape}")
                else:
                    print(f"Output shape: {output.shape}")
        
        print(f"✓ {model_name} test passed")
        return True
        
    except Exception as e:
        print(f"✗ {model_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all model types."""
    print("Testing all MLP models...")
    
    # Test parameters
    batch_size = 4  # Use batch size > 1 to avoid BatchNorm issues
    input_size = 784  # MNIST flattened
    num_classes = 10
    time_steps = 8  # Smaller for testing
    
    # Create test inputs
    static_input = torch.randn(batch_size, input_size)  # [batch, features]
    snn_input = torch.randn(time_steps, batch_size, input_size)  # [time, batch, features]
    
    results = {}
    
    # Test Standard MLP
    results['MLP'] = test_model(
        MLP, "Standard MLP", static_input,
        num_classes=num_classes, input_size=input_size
    )
    
    # Test SNN MLP
    results['SNN_MLP'] = test_model(
        MLPSNN, "SNN MLP", snn_input,
        num_classes=num_classes, input_size=input_size, time_steps=time_steps
    )
    
    # Test Branchy MLP
    results['Branchy_MLP'] = test_model(
        BranchyMLP, "Branchy MLP", static_input,
        num_classes=num_classes, input_size=input_size
    )
    
    # Test Quantized SNN MLP
    results['Quantized_SNN_MLP'] = test_model(
        QuantizedSNNMLP, "Quantized SNN MLP", snn_input,
        num_classes=num_classes, input_size=input_size, time_steps=time_steps
    )
    
    # Test Branchy Quantized SNN MLP
    results['Branchy_Quantized_SNN_MLP'] = test_model(
        BranchyQuantizedSNNMLP, "Branchy Quantized SNN MLP", static_input,
        num_classes=num_classes, input_size=input_size
    )
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name:25} {status}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All model tests passed! Ready for training.")
    else:
        print("⚠️  Some models failed. Check errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 