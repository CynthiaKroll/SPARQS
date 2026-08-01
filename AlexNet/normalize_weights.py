import torch
import argparse
import os

# Project specific imports
import data_loader
import utils
from models.alexnet_qcfs import AlexNetQCFS # Import the QCFS model definition
from config import config # To get num_classes if needed

def get_device():
    """Gets the appropriate device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Normalize weights of a trained QCFS model and calculate SNN thresholds.")
    parser.add_argument('--input_weights', type=str, required=True,
                        help='Path to the trained QCFS model weights file (e.g., results/alexnet_qcfs_ste_trained.pth).')
    parser.add_argument('--output_weights', type=str, required=True,
                        help='Path to save the normalized weights file (e.g., results/alexnet_qcfs_ste_normalized.pth).')
    parser.add_argument('--output_thresholds', type=str, required=True,
                        help='Path to save the calculated SNN thresholds (e.g., results/snn_thresholds.pth).')
    parser.add_argument('--calib_batches', type=int, default=10, # Use a subset for faster calibration
                        help='Number of batches from training set to use for calibration.')
    parser.add_argument('--time_steps', type=int, default=20,
                        help='Number of time steps (T) used for SNN threshold calculation.')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # --- 1. Load Model Structure ---
    print("Loading AlexNetQCFS model structure...")
    # Ensure num_classes matches the trained model, get from config or default
    num_classes = config.get("num_classes", 10)
    model = AlexNetQCFS(num_classes=num_classes)
    model.to(device)
    model.eval() # Set to evaluation mode

    # --- 2. Load Trained Weights ---
    if not os.path.exists(args.input_weights):
        print(f"Error: Input weights file not found at {args.input_weights}")
        return

    print(f"Loading trained weights from: {args.input_weights}")
    # Use weights_only=True for security if loading untrusted files
    # state_dict = torch.load(args.input_weights, map_location=device, weights_only=True)
    # For this project, assuming weights are trusted
    state_dict = torch.load(args.input_weights, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Trained weights loaded successfully.")

    # --- 3. Get Calibration Loader ---
    print(f"Getting calibration data loader ({args.calib_batches} batches)...")
    # Use a limited number of batches for calibration as specified
    calib_loader = data_loader.get_calibration_loader(num_batches=args.calib_batches)
    print(f"Calibration loader ready with {len(calib_loader)} batches.")


    # --- 4. Normalize Weights and Calculate Thresholds ---
    print("Starting weight normalization and threshold calculation...")
    # The function modifies the model's weights in-place and returns thresholds
    snn_thresholds = utils.normalize_weights(model, calib_loader, time_steps=args.time_steps)
    print("Weight normalization and threshold calculation finished.")

    # --- 5. Save Normalized Weights ---
    # Ensure output directory exists
    output_weights_dir = os.path.dirname(args.output_weights)
    if output_weights_dir: # Check if output_dir is not empty
        os.makedirs(output_weights_dir, exist_ok=True)

    print(f"Saving normalized weights to: {args.output_weights}")
    # Save the state_dict of the modified model
    torch.save(model.state_dict(), args.output_weights)
    print("Normalized weights saved successfully.")

    # --- 6. Save SNN Thresholds ---
    output_thresholds_dir = os.path.dirname(args.output_thresholds)
    if output_thresholds_dir:
        os.makedirs(output_thresholds_dir, exist_ok=True)

    print(f"Saving SNN thresholds to: {args.output_thresholds}")
    torch.save(snn_thresholds, args.output_thresholds)
    print("SNN thresholds saved successfully.")

if __name__ == '__main__':
    main()