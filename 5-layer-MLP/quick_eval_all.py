import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS ({elapsed:.2f}s): {description}")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED ({elapsed:.2f}s): {description}")
        print("STDERR:", e.stderr[-500:] if e.stderr else "No error output")
        print("STDOUT:", e.stdout[-500:] if e.stdout else "No standard output")
        return False

def main():
    """Main evaluation function."""
    print("🚀 Starting comprehensive 5-layer MLP evaluation...")
    print("This will evaluate all model variants following AlexNet patterns")
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Create results directories
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Define models to evaluate (matching AlexNet structure)
    # Time steps must match how each model was TRAINED (run_training.sh):
    # the standard SNN uses T=32, the quantized pair (QSNN/QDSNN) uses T=4.
    models_to_evaluate = [
        ("standard_mlp", "Standard MLP", "--model standard_mlp"),
        ("snn_mlp", "SNN MLP", "--model snn_mlp --time_steps 32"),
        ("branchy_mlp", "Branchy MLP", "--model branchy_mlp"),
        ("quantized_snn_mlp", "Quantized SNN MLP", "--model quantized_snn_mlp --time_steps 4"),
        ("branchy_quantized_snn_mlp", "Branchy Quantized SNN MLP", "--model branchy_quantized_snn_mlp --time_steps 4"),
    ]
    
    successful_evaluations = []
    failed_evaluations = []
    
    # Check if model files exist
    model_files = {
        "standard_mlp": "results/standard_mlp.pth",
        "snn_mlp": "results/snn_mlp.pth", 
        "branchy_mlp": "results/branchy_mlp.pth",
        "quantized_snn_mlp": "results/quantized_snn_mlp.pth",
        "branchy_quantized_snn_mlp": "results/branchy_quantized_snn_mlp.pth",
    }
    
    print("\n📋 Model file status:")
    for model_key, model_path in model_files.items():
        if os.path.exists(model_path):
            print(f"✅ {model_key}: {model_path}")
        else:
            print(f"❌ {model_key}: {model_path} (missing)")
    
    # Evaluate each model
    for model_key, description, eval_args in models_to_evaluate:
        print(f"\n🔍 Evaluating: {description}")
        
        # Check if model file exists
        model_path = model_files.get(model_key)
        if not model_path or not os.path.exists(model_path):
            print(f"⚠️  Model file {model_path} not found, skipping {description}")
            failed_evaluations.append((model_key, "Model file not found"))
            continue
        
        # Run evaluation
        cmd = f"python evaluate.py {eval_args}"
        
        success = run_command(cmd, f"Evaluate {description}")
        
        if success:
            successful_evaluations.append(model_key)
            # Check if CSV was created
            csv_path = f"results/logs/{model_key}_eval.csv"
            if os.path.exists(csv_path):
                print(f"✅ Results saved to {csv_path}")
            else:
                print(f"⚠️  Expected CSV file {csv_path} not found")
        else:
            failed_evaluations.append((model_key, "Evaluation command failed"))
    
    # Generate visualizations if we have any successful evaluations
    if successful_evaluations:
        print(f"\n📊 Generating visualizations for {len(successful_evaluations)} models...")
        viz_success = run_command("python visualize.py", "Generate visualizations")
        
        if viz_success:
            print("✅ Visualizations generated successfully")
            # List generated plots
            plots_dir = "results/plots"
            if os.path.exists(plots_dir):
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                print(f"📈 Generated {len(plot_files)} plots:")
                for plot_file in sorted(plot_files):
                    print(f"   - {plot_file}")
        else:
            print("❌ Visualization generation failed")
    else:
        print("❌ No successful evaluations, skipping visualization generation")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful evaluations: {len(successful_evaluations)}")
    for model in successful_evaluations:
        print(f"   - {model}")
    
    if failed_evaluations:
        print(f"❌ Failed evaluations: {len(failed_evaluations)}")
        for model, reason in failed_evaluations:
            print(f"   - {model}: {reason}")
    
    # Check results
    print(f"\n📁 Results location:")
    print(f"   - CSV files: results/logs/")
    print(f"   - Plot files: results/plots/")
    
    csv_files = []
    if os.path.exists("results/logs"):
        csv_files = [f for f in os.listdir("results/logs") if f.endswith('_eval.csv')]
        print(f"   - Found {len(csv_files)} CSV files")
    
    plot_files = []
    if os.path.exists("results/plots"):
        plot_files = [f for f in os.listdir("results/plots") if f.endswith('.png')]
        print(f"   - Found {len(plot_files)} plot files")
    
    if successful_evaluations:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"   You can find detailed results in results/logs/ and plots in results/plots/")
        return 0
    else:
        print(f"\n💥 All evaluations failed!")
        print(f"   Please check that model files exist and training completed successfully")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 