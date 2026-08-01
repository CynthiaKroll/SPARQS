import argparse
import torch
import time
import os
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json
from collections import defaultdict

from models.alexnet import AlexNet
from models.branchy_models import BranchyAlexNet
from models.branchy_quantized_snn import BranchyQuantizedSNN
from models.rl_environment import QLearningAgent
import pickle
from config import config
from data_loader import get_data_loaders
from utils import calculate_energy, calculate_inference_time, calculate_power

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_branchy_model(model_name, model_path, device):
    """Load a branchy model and return it"""
    print(f"Loading model: {model_name}")
    
    if "branchy_quantized_snn" in model_name:
        # Load quantized branchy model
        model = BranchyQuantizedSNN(num_classes=10)
        
        # Load the state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        
        # Load RL agent if available
        agent_path = model_path.replace('.pth', '_agent.pkl')
        if os.path.exists(agent_path):
            try:
                with open(agent_path, 'rb') as f:
                    agent = pickle.load(f)
                model.rl_agent = agent
                print(f"Loaded RL agent from {agent_path}")
            except Exception as e:
                print(f"Failed to load RL agent: {e}")
        
        # Set quantization and move to CPU for quantized models
        if hasattr(model, 'is_quantized'):
            model.is_quantized = True
        model.to('cpu')  # Quantized models run on CPU
        
    elif "dynamic_alexnet" in model_name or "branchy" in model_name:
        # Load regular branchy model
        model = BranchyAlexNet(num_classes=10)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model.eval()
    return model

def evaluate_exit_distribution(model, model_name, test_loader, device, thresholds=None):
    """
    Evaluate a branchy model and track exit distribution with class information
    
    Returns:
        dict: Contains accuracy, exit distribution, and per-exit class statistics
    """
    model.eval()
    
    # Initialize tracking variables
    total_correct = 0
    total_samples = 0
    exit_counts = defaultdict(int)
    per_exit_correct = defaultdict(int)
    per_exit_samples = defaultdict(int)
    exit_decisions = []  # Store exit decisions for each sample
    sample_labels = []   # Store true labels for each sample
    
    # Track class distributions per exit
    exit_class_counts = defaultdict(lambda: defaultdict(int))  # exit_idx -> class_idx -> count
    
    # Set up thresholds if provided
    if thresholds and hasattr(model, 'rl_agent') and model.rl_agent:
        if hasattr(model.rl_agent, 'confidence_thresholds'):
            model.rl_agent.confidence_thresholds = thresholds
        model.confidence_thresholds = thresholds
        print(f"Using thresholds: {thresholds}")
    
    print(f"Evaluating {model_name} for exit distribution...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(enumerate(test_loader), desc="Evaluating"):
            # Move to appropriate device
            if "quantized_snn" in model_name:
                images, labels = images.to('cpu'), labels.to('cpu')
            else:
                images, labels = images.to(device), labels.to(device)
            
            batch_size = images.size(0)
            
            # Forward pass
            model_output = model(images)
            
            # Parse model output for branchy models
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                exit_logits_list = model_output[0]  # List of exit outputs
                final_logits = model_output[1]      # Final output
                
                # Get predictions and confidences for each exit
                exit_probs = []
                exit_preds = []
                exit_confs = []
                
                for exit_logits in exit_logits_list:
                    # Apply temperature scaling if available
                    if hasattr(model, 'temperature1') and len(exit_probs) == 0:
                        exit_logits = exit_logits / model.temperature1
                    elif hasattr(model, 'temperature2') and len(exit_probs) == 1:
                        exit_logits = exit_logits / model.temperature2
                    
                    probs = torch.softmax(exit_logits.float(), dim=1)
                    conf, pred = probs.max(dim=1)
                    
                    exit_probs.append(probs)
                    exit_preds.append(pred)
                    exit_confs.append(conf)
                
                # Final exit
                final_probs = torch.softmax(final_logits.float(), dim=1)
                final_pred = final_probs.argmax(dim=1)
                
                # Determine exit decisions
                batch_exit_decisions = torch.full((batch_size,), len(exit_preds), dtype=torch.long)  # Default to final exit
                final_predictions = torch.zeros_like(labels)
                
                # Apply early exit logic
                if thresholds and len(thresholds) == len(exit_confs):
                    remaining_mask = torch.ones(batch_size, dtype=torch.bool)
                    
                    for exit_idx, (conf, pred, threshold) in enumerate(zip(exit_confs, exit_preds, thresholds)):
                        # Check which samples should exit at this stage
                        exit_mask = (conf >= threshold) & remaining_mask
                        
                        if exit_mask.any():
                            batch_exit_decisions[exit_mask] = exit_idx
                            final_predictions[exit_mask] = pred[exit_mask]
                            remaining_mask[exit_mask] = False
                    
                    # Remaining samples go to final exit
                    if remaining_mask.any():
                        final_predictions[remaining_mask] = final_pred[remaining_mask]
                        
                else:
                    # No threshold-based early exit, use final predictions
                    final_predictions = final_pred
                
                # Track per-exit statistics and class distributions
                for exit_idx in range(len(exit_preds) + 1):  # +1 for final exit
                    exit_mask = (batch_exit_decisions == exit_idx)
                    if exit_mask.any():
                        exit_counts[exit_idx] += exit_mask.sum().item()
                        per_exit_samples[exit_idx] += exit_mask.sum().item()
                        
                        # Track class distribution for this exit
                        exit_labels = labels[exit_mask]
                        for class_idx in range(10):  # CIFAR-10 has 10 classes
                            class_count = (exit_labels == class_idx).sum().item()
                            exit_class_counts[exit_idx][class_idx] += class_count
                        
                        if exit_idx < len(exit_preds):
                            # Early exit
                            correct = (exit_preds[exit_idx][exit_mask] == labels[exit_mask]).sum().item()
                        else:
                            # Final exit
                            correct = (final_pred[exit_mask] == labels[exit_mask]).sum().item()
                        
                        per_exit_correct[exit_idx] += correct
                
                # Overall accuracy
                total_correct += (final_predictions == labels).sum().item()
                total_samples += batch_size
                
                # Store exit decisions and labels for this batch
                exit_decisions.extend(batch_exit_decisions.cpu().numpy())
                sample_labels.extend(labels.cpu().numpy())
                
            else:
                # Not a branchy model, treat as single exit
                if isinstance(model_output, torch.Tensor):
                    predictions = model_output.argmax(dim=1)
                else:
                    predictions = model_output[-1].argmax(dim=1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += batch_size
                exit_counts[0] += batch_size
                per_exit_correct[0] += (predictions == labels).sum().item()
                per_exit_samples[0] += batch_size
                exit_decisions.extend([0] * batch_size)
                sample_labels.extend(labels.cpu().numpy())
                
                # Track class distribution for single exit
                for class_idx in range(10):
                    class_count = (labels == class_idx).sum().item()
                    exit_class_counts[0][class_idx] += class_count
    
    # Calculate statistics
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    per_exit_accuracy = {}
    for exit_idx in per_exit_samples:
        if per_exit_samples[exit_idx] > 0:
            per_exit_accuracy[exit_idx] = per_exit_correct[exit_idx] / per_exit_samples[exit_idx]
        else:
            per_exit_accuracy[exit_idx] = 0.0
    
    # Convert to percentages for distribution
    exit_distribution = {}
    for exit_idx in exit_counts:
        exit_distribution[exit_idx] = (exit_counts[exit_idx] / total_samples) * 100 if total_samples > 0 else 0.0
    
    # Convert class counts to percentages for each exit
    exit_class_distributions = {}
    for exit_idx in exit_class_counts:
        total_exit_samples = sum(exit_class_counts[exit_idx].values())
        if total_exit_samples > 0:
            exit_class_distributions[exit_idx] = {
                class_idx: (count / total_exit_samples) * 100 
                for class_idx, count in exit_class_counts[exit_idx].items()
            }
        else:
            exit_class_distributions[exit_idx] = {class_idx: 0.0 for class_idx in range(10)}
    
    results = {
        'model_name': model_name,
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'exit_counts': dict(exit_counts),
        'exit_distribution': exit_distribution,
        'per_exit_accuracy': per_exit_accuracy,
        'exit_decisions': exit_decisions,
        'sample_labels': sample_labels,
        'exit_class_counts': dict(exit_class_counts),
        'exit_class_distributions': exit_class_distributions,
        'thresholds': thresholds
    }
    
    return results

def plot_exit_distribution(results_list, save_path="results/plots/early_exit_distribution.png"):
    """Plot early exit distribution for multiple models"""
    
    # Set up the plot with LaTeX-compatible styling
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
        'mathtext.fontset': 'stix',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })
    
    # Model display names
    MODEL_DISPLAY_NAMES = {
        "dynamic_alexnet": "Dynamic AlexNet",
        "branchy_quantized_snn_th0.6_0.7": "QDSNN (T=0.6/0.7)",
        "branchy_quantized_snn_th0.2_0.4": "QDSNN (T=0.2/0.4)",
        "branchy_quantized_snn": "QDSNN"
    }
    
    # Prepare data for plotting
    models = []
    exit_names = ["Exit 1", "Exit 2", "Final Exit"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    # Calculate bar positions
    bar_width = 0.8
    y_positions = []
    model_labels = []
    
    for i, result in enumerate(results_list):
        model_name = result['model_name']
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        model_labels.append(display_name)
        y_positions.append(i)
        
        # Get exit distribution
        distribution = result['exit_distribution']
        
        # Create stacked horizontal bars
        left = 0
        for exit_idx, color in enumerate(colors):
            if exit_idx in distribution:
                width = distribution[exit_idx]
                if width > 0:
                    bar = ax.barh(i, width, left=left, height=bar_width, 
                                color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # Add percentage labels if significant
                    if width > 5:  # Only label if > 5%
                        ax.text(left + width/2, i, f'{width:.1f}%', 
                               ha='center', va='center', fontweight='bold', 
                               fontsize=12, color='white' if width > 15 else 'black')
                    
                    left += width
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_labels)
    ax.set_xlabel('Percentage of Samples (%)', fontweight='bold', fontsize=16)
    ax.set_title('Early Exit Distribution Across Models', fontweight='bold', fontsize=18)
    ax.set_xlim(0, 100)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                      for color in colors]
    ax.legend(legend_elements, exit_names, loc='lower right', frameon=True, 
             fancybox=True, shadow=True, fontsize=14)
    
    # Style the axes
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.grid(True, alpha=0.4, linewidth=0.8, axis='x')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Early exit distribution plot saved to {save_path}")

def plot_class_distribution_per_exit(results_list, save_path="results/plots/class_distribution_per_exit.png"):
    """Plot class distribution for each exit point across all models"""
    
    # Set up the plot with LaTeX-compatible styling
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
        'mathtext.fontset': 'stix',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })
    
    # Find all unique exit indices across all models
    all_exit_indices = set()
    for result in results_list:
        all_exit_indices.update(result['exit_class_distributions'].keys())
    
    max_exits = max(len(all_exit_indices), 1)
    sorted_exit_indices = sorted(all_exit_indices)
    
    # Create subplot grid
    fig, axes = plt.subplots(len(results_list), max_exits, figsize=(6*max_exits, 6*len(results_list)), dpi=300)
    
    # Handle single subplot case
    if len(results_list) == 1 and max_exits == 1:
        axes = np.array([[axes]])
    elif len(results_list) == 1:
        axes = axes.reshape(1, -1)
    elif max_exits == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each model
    for model_idx, result in enumerate(results_list):
        model_name = result['model_name']
        exit_class_distributions = result['exit_class_distributions']
        
        # Model display names
        MODEL_DISPLAY_NAMES = {
            "dynamic_alexnet": "Dynamic AlexNet",
            "branchy_quantized_snn_th0.6_0.7": "QDSNN (T=0.6/0.7)",
            "branchy_quantized_snn_th0.2_0.4": "QDSNN (T=0.2/0.4)",
            "branchy_quantized_snn": "QDSNN"
        }
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        # Plot class distribution for exits that exist for this model
        for col_idx, exit_idx in enumerate(sorted_exit_indices):
            if col_idx >= max_exits:
                break
                
            ax = axes[model_idx, col_idx]
            
            if exit_idx in exit_class_distributions:
                class_percentages = []
                for class_idx in range(10):
                    percentage = exit_class_distributions[exit_idx].get(class_idx, 0.0)
                    class_percentages.append(percentage)
                
                # Create bar plot
                bars = ax.bar(CIFAR10_CLASSES, class_percentages, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add percentage labels on bars
                for bar, percentage in zip(bars, class_percentages):
                    if percentage > 0.5:  # Only show labels for bars with significant height
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                               f'{percentage:.1f}%', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
                
                # Customize the subplot
                ax.set_ylabel('Percentage of Samples (%)', fontweight='bold')
                ax.set_xlabel('Class Label', fontweight='bold')
                
                # Set title based on exit type
                if exit_idx == max(exit_class_distributions.keys()):
                    exit_name = f"Final Classifier ({exit_idx+1})"
                else:
                    exit_name = f"Early Exit ({exit_idx+1})"
                
                ax.set_title(f"{display_name} - {exit_name}", fontweight='bold')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
                # Style the axes
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                ax.grid(True, alpha=0.4, linewidth=0.8, axis='y')
                
                # Set consistent y-axis limits
                ax.set_ylim(0, max(max(class_percentages) * 1.1, 10))
            else:
                # Hide subplot if this exit doesn't exist for this model
                ax.set_visible(False)
    
    # Hide unused subplots if any
    if len(results_list) < axes.shape[0]:
        for model_idx in range(len(results_list), axes.shape[0]):
            for col_idx in range(axes.shape[1]):
                axes[model_idx, col_idx].set_visible(False)
    
    # Add overall title
    fig.suptitle('CIFAR-10 - Class Distributions Across Exits', fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")

def save_exit_distribution_data(results_list, save_path="results/logs/early_exit_distribution.csv"):
    """Save exit distribution data to CSV"""
    
    # Prepare data for CSV
    csv_data = []
    
    for result in results_list:
        model_name = result['model_name']
        distribution = result['exit_distribution']
        per_exit_acc = result['per_exit_accuracy']
        thresholds = result.get('thresholds', [])
        exit_class_distributions = result.get('exit_class_distributions', {})
        
        row = {
            'model_name': model_name,
            'overall_accuracy': result['overall_accuracy'],
            'total_samples': result['total_samples'],
            'thresholds': str(thresholds) if thresholds else 'N/A'
        }
        
        # Add exit distribution percentages
        for exit_idx in range(3):  # Assume max 3 exits (2 early + 1 final)
            row[f'exit_{exit_idx}_percentage'] = distribution.get(exit_idx, 0.0)
            row[f'exit_{exit_idx}_accuracy'] = per_exit_acc.get(exit_idx, 0.0)
            row[f'exit_{exit_idx}_count'] = result['exit_counts'].get(exit_idx, 0)
            
            # Add class distribution for this exit
            if exit_idx in exit_class_distributions:
                for class_idx, class_name in enumerate(CIFAR10_CLASSES):
                    class_percentage = exit_class_distributions[exit_idx].get(class_idx, 0.0)
                    row[f'exit_{exit_idx}_class_{class_name}_percentage'] = class_percentage
        
        csv_data.append(row)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(save_path, index=False)
    
    print(f"Exit distribution data saved to {save_path}")
    print("\nExit Distribution Summary:")
    print(df[['model_name', 'overall_accuracy', 'exit_0_percentage', 'exit_1_percentage', 'exit_2_percentage']].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Early Exit Distribution")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--models', nargs='+', required=True, 
                       help='Model names to evaluate (e.g., dynamic_alexnet branchy_quantized_snn_th0.6_0.7)')
    parser.add_argument('--model_paths', nargs='+', required=True,
                       help='Paths to model files')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.6, 0.7],
                       help='Confidence thresholds for early exit')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Load test data
    _, _, test_loader, _ = get_data_loaders()
    
    # Evaluate all models
    results_list = []
    
    for model_name, model_path in zip(args.models, args.model_paths):
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, skipping {model_name}")
            continue
        
        try:
            # Load model
            model = load_branchy_model(model_name, model_path, device)
            
            # Evaluate with thresholds
            thresholds_to_use = args.thresholds if "branchy" in model_name else None
            result = evaluate_exit_distribution(model, model_name, test_loader, device, thresholds_to_use)
            results_list.append(result)
            
            print(f"\n{model_name} Results:")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.3f}")
            print(f"  Exit Distribution: {result['exit_distribution']}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    if results_list:
        # Plot and save results
        plot_exit_distribution(results_list)
        plot_class_distribution_per_exit(results_list)
        save_exit_distribution_data(results_list)
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main() 