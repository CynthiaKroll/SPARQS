#!/usr/bin/env python3
"""Quick evaluation of all available LeNet models with corrected SNN handling."""

import os
import sys
import subprocess

def run_evaluation(model_name, model_path):
    """Run evaluation for a single model."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"🔍 Evaluating {model_name}...")
    try:
        cmd = [sys.executable, "evaluate.py", "--model", model_name, "--model_path", model_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {model_name} evaluation completed")
            return True
        else:
            print(f"❌ {model_name} evaluation failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} evaluation timed out")
        return False
    except Exception as e:
        print(f"❌ {model_name} evaluation error: {e}")
        return False

def main():
    """Evaluate all available models."""
    print("🚀 Starting comprehensive LeNet model evaluation...")
    
    # FP32 baselines only -- these go through the legacy per-model evaluate.py.
    # The quantized pair (QSNN/QDSNN) is measured by evaluate_quantized_pair.py
    # (see below); it cannot be evaluated via evaluate.py (wrong encoder + no
    # INT8 load).
    models_to_eval = [
        ("lenet", "results/standard_lenet.pth"),
        ("dynamic_lenet", "results/branchy_lenet.pth"),
        ("snn_lenet", "results/lenet_snn.pth"),
    ]

    # Alternative paths to check
    alt_paths = {
        "lenet": ["results/lenet.pth"],
        "snn_lenet": ["results/snn_lenet.pth"],
    }
    
    success_count = 0
    total_count = 0
    
    for model_name, model_path in models_to_eval:
        total_count += 1
        
        # Check alternative paths if main path doesn't exist
        if not os.path.exists(model_path) and model_name in alt_paths:
            for alt_path in alt_paths[model_name]:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
        
        result = run_evaluation(model_name, model_path)
        if result:
            success_count += 1

    # Quantized pair (QSNN vs QDSNN) via the unified evaluator.
    if os.path.exists("results/quantized_snn_lenet_int8.pth"):
        print("🔍 Evaluating quantized pair (QSNN vs QDSNN)...")
        subprocess.run([sys.executable, "evaluate_quantized_pair.py"], timeout=1800)

    print(f"\n📊 Evaluation Summary: {success_count}/{total_count} FP32 models evaluated successfully")
    
    # Show results summary
    print("\n📈 Results Summary:")
    print("-" * 50)
    
    logs_dir = "results/logs"
    if os.path.exists(logs_dir):
        import pandas as pd
        for csv_file in sorted(os.listdir(logs_dir)):
            if csv_file.endswith('_eval.csv'):
                try:
                    df = pd.read_csv(os.path.join(logs_dir, csv_file))
                    if not df.empty:
                        row = df.iloc[-1]
                        model_name = row.get('model_name', 'Unknown')
                        accuracy = row.get('accuracy', 0) * 100
                        energy = row.get('energy', 0)
                        power = row.get('power', 0)
                        print(f"{model_name:20s}: {accuracy:6.2f}% acc, {energy:8.2e} J, {power:6.3f} W")
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    
    print("\n🎯 All evaluations completed!")

if __name__ == "__main__":
    main() 