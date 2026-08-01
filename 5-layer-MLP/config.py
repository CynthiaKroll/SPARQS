import numpy as np

# --- Theoretical Energy Constants ---
# Memory access energy per byte.
# The SPARQ paper (Eq. 4 and Eq. 5) and the root README both specify
# E_mem = 80e-12 J/byte (Horowitz 2014). That is the value used here so the energy
# results follow the published methodology.
E_MEM_J_PER_BYTE_PAPER = 80e-12     # 80 pJ/Byte - value used by the SPARQ paper
E_MEM_J_PER_BYTE_DDR3_45NM = 4e-12  # 4 pJ/Byte - DDR3 45nm (kept for reference only)

# Standard energy per operation constants
E_MAC_32BIT_J = 4.6e-12  # Energy for a 32-bit floating-point Multiply-Accumulate
E_MAC_8BIT_J = 0.2e-12   # Energy for an 8-bit integer Multiply-Accumulate  
E_AC_8BIT_J = 0.03e-12   # Energy for an 8-bit integer Accumulate (Add) - typical for SNNs

# Memory access energy (paper value; see above)
E_MEM_J_PER_BYTE = E_MEM_J_PER_BYTE_PAPER

# Neuromorphic energy constants
E_AC_NEUROMORPHIC_J = 0.01e-12     # 0.01 pJ per AC - more realistic for neuromorphic
E_MEM_NEUROMORPHIC_J_PER_BYTE = 2e-12  # 2 pJ per Byte - more realistic neuromorphic memory

config = {
    # Dataset configuration (MNIST for MLP)
    "dataset": "MNIST",
    "num_classes": 10,
    "image_size": (1, 28, 28),  # MNIST: 1 channel, 28x28 pixels
    "input_size": 1 * 28 * 28,  # Flattened input size for MLP
    "batch_size": 128,
    "test_batch_size": 1000,
    
    # Training configuration
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "early_stopping_patience": 10,
    
    # SNN-specific training configuration. NOTE: applies to the FP32 SNN
    # baseline (train_snn_mlp) only; the quantized pair (QSNN/QDSNN) trains at
    # the base LR 0.001, unified across all three modules.
    "snn_learning_rate_multiplier": 5.0,
    
    # SNN configuration
    "snn": {
        "beta": 0.95,           # Membrane decay rate
        "threshold": 1.0,       # Firing threshold
        "spike_grad": "ATan",   # Surrogate gradient function
        "reset_mechanism": "subtract"
    },
    
    # Conversion settings (matching AlexNet)
    "conversion": {
        "calibration_samples": 1000,
        "momentum": 1.0,
        "mode": "99.9%"
    },
    
    # Time steps for SNN simulation
    "num_steps": 32,   # Optimal time window for energy efficiency (was 150)
    
    # Quantization settings
    "quantization": {
        "bits": 8,
        # Engine is auto-selected at runtime (get_quant_engine): fbgemm ->
        # qnnpack -> onednn. This box only exposes 'onednn'. Do NOT hardcode.
        "backend": "auto",
        "calibration_batches": 10,
        "qconfig_strategy": "default"
    },
    
    # Hardware profiles with realistic energy constants
    "hardware_constants": {
        "a100": {
            "energy_per_mac": E_MAC_32BIT_J,  # 4.6 pJ per MAC
            "energy_per_spike": 0.1e-12,      # 0.1 pJ per spike
            "energy_per_ac": E_AC_8BIT_J,     # 0.03 pJ per AC
            "energy_per_mem_access": E_MEM_J_PER_BYTE,  # 4 pJ per Byte
            "gpu_flops": 312e12  # 312 TFLOPS (FP32)
        },
        "v100": {
            "energy_per_mac": E_MAC_32BIT_J,  # 4.6 pJ per MAC
            "energy_per_spike": 0.1e-12,      # 0.1 pJ per spike
            "energy_per_ac": E_AC_8BIT_J,     # 0.03 pJ per AC
            "energy_per_mem_access": E_MEM_J_PER_BYTE,  # 4 pJ per Byte
            "gpu_flops": 125e12  # 125 TFLOPS (FP32)
        },
        "rtx3090": {
            "energy_per_mac": E_MAC_32BIT_J,  # 4.6 pJ per MAC
            "energy_per_spike": 0.1e-12,      # 0.1 pJ per spike
            "energy_per_ac": E_AC_8BIT_J,     # 0.03 pJ per AC
            "energy_per_mem_access": E_MEM_J_PER_BYTE,  # 4 pJ per Byte
            "gpu_flops": 35.6e12  # 35.6 TFLOPS (FP32)
        },
        "neuromorphic": {
            "energy_per_mac": 10.0e-12, # 10 pJ per MAC (less relevant for SNNs)
            "energy_per_spike": 0.01e-12, # 0.01 pJ per spike - more realistic for neuromorphic
            "energy_per_ac": 0.01e-12,    # 0.01 pJ per AC - more realistic for neuromorphic
            "energy_per_mem_access": 2e-12, # 2 pJ per Byte - more realistic neuromorphic memory
            "gpu_flops": 5e12  # 5 TFLOPS (less relevant)
        }
    },
    
    # Default hardware profile
    "hardware_profile": "a100",
    
    # Early exit configuration for branchy models
    "early_exit": {
        "threshold_1": 0.9,   # Confidence threshold for first exit
        "threshold_2": 0.8,   # Confidence threshold for second exit
        "use_temperature_scaling": True,
        "temperature_1": 1.0,
        "temperature_2": 1.0
    },
    
    # Mixup data augmentation (paper, Sec. III-C). alpha=0 disables it.
    "mixup_alpha": 0.4,

    # Reinforcement Learning configuration for dynamic early exit.
    # Values follow the paper (Sec. III-B): eta=0.1, gamma=0.9, alpha=0.3.
    "rl": {
        "n_confidence_bins": 10,
        "learning_rate": 0.1,       # eta
        "epsilon": 0.5,             # initial epsilon-greedy exploration rate (decayed)
        "gamma": 0.9,               # discount factor (paper says 0.9, was 0.95)
        "reward_scale": 0.3,        # alpha in the reward function (paper Eq. 1)
        "episodes": 1000
    },
    
    # Paths
    "data_path": "./data",
    "results_path": "./results",
    "logs_path": "./results/logs",
    "plots_path": "./results/plots",
    
    # Model configurations
    "models": {
        "mlp": {
            "type": "MLP",
            "input_size": 1 * 28 * 28,
            "num_classes": 10
        },
        "snn_mlp": {
            "type": "SNNMLP", 
            "input_size": 1 * 28 * 28,
            "num_classes": 10,
            "time_steps": 32
        },
        "branchy_mlp": {
            "type": "BranchyMLP",
            "input_size": 1 * 28 * 28,
            "num_classes": 10,
            "num_exits": 2
        }
    },
    
    # Visualization settings
    "visualization": {
        "dpi": 300,
        "figsize": (12, 8),
        "style": "seaborn-v0_8",
        "colors": {
            "standard": "#1f77b4",
            "snn": "#ff7f0e", 
            "quantized": "#2ca02c",
            "branchy": "#d62728",
            "dynamic": "#9467bd"
        }
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "mlp_experiments.log"
    }
}
