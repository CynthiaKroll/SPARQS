
import numpy as np

# The SPARQ paper (Eq. 4 and Eq. 5) specifies E_mem = 80e-12 J/byte
# (Horowitz 2014), the same value the 5-layer-MLP module uses.
E_MEM_J_PER_BYTE_PAPER = 80e-12     # 80 pJ/Byte - value used by the SPARQ paper
E_MEM_J_PER_BYTE_DDR3_45NM = 4e-12  # 4 pJ/Byte - DDR3 45nm (kept for reference only)

E_MAC_32BIT_J = 4.6e-12  # Energy for a 32-bit floating-point Multiply-Accumulate
E_MAC_8BIT_J = 0.2e-12   # Energy for an 8-bit integer Multiply-Accumulate
E_AC_8BIT_J = 0.03e-12   # Energy for an 8-bit integer Accumulate

E_MEM_J_PER_BYTE = E_MEM_J_PER_BYTE_PAPER

E_AC_NEUROMORPHIC_J = 0.01e-12     # 0.01 pJ per AC
E_MEM_NEUROMORPHIC_J_PER_BYTE = 2e-12  # 2 pJ per Byte

# The two SNN time windows, mirroring the 5-layer-MLP scheme (its
# run_training.sh SNN_T=32 / QNT_T=4). These are the single source of truth:
# training, both evaluators, and the plot labels all derive T from here.
#
# The split is principled, not arbitrary: a plain FP32 SNN needs a longer rate
# window to reach good accuracy, whereas the QAT-trained QSNN/QDSNN hold
# accuracy at far fewer steps. Running the quantized pair at T=4 is what makes
# their energy competitive -- energy is ~100% weight-memory traffic and scales
# ~linearly with T, so 4 vs 32 is an ~8x swing.
STANDARD_SNN_STEPS = 32   # FP32 SNN baseline (snn_lenet)
QUANTIZED_SNN_STEPS = 4   # QAT pair (quantized_snn_lenet, branchy_quantized_snn)

config = {
    "dataset": "MNIST",
    "num_classes": 10,
    "image_size": (1, 28, 28),  
    "batch_size": 128,
    "test_batch_size": 1000,
    
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "early_stopping_patience": 10,
    
    "snn": {
        "beta": 0.95,           
        "threshold": 1.0,       
        "spike_grad": "ATan",   
        "reset_mechanism": "subtract"
    },
    
    "conversion": {
        "calibration_samples": 1000,
        "momentum": 1.0,
        "mode": "99.9%"
    },
    
    "num_steps": STANDARD_SNN_STEPS,          # default / FP32 SNN window
    "quantized_num_steps": QUANTIZED_SNN_STEPS,  # QAT pair window
    # Gain for Poisson rate encoding: P(spike per step) = clamp(x_norm*gain, 0, 1),
    # x_norm min-max normalized. gain=1.0 matches the MLP module's canonical
    # encoder (gain=2.0 saturates: every pixel above 0.5 fires on EVERY step).
    # Do NOT divide by num_steps -- that destroys the rate code.
    "poisson_gain": 1.0,

    # Mixup data augmentation (paper Sec. III-C). alpha=0 disables it.
    "mixup_alpha": 0.4,

    "quantization": {
        "bits": 8,
        # Engine is auto-selected at runtime (get_quant_engine): fbgemm -> qnnpack
        # -> onednn. This box only exposes 'onednn', so a hardcoded 'fbgemm' made
        # prepare_qat/convert fail. Do NOT hardcode an engine here.
        "backend": "auto",
        "calibration_batches": 10,
        "qconfig_strategy": "default"
    },
    
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
            "energy_per_mac": 10.0e-12, # 10 pJ per MAC 
            "energy_per_spike": 0.01e-12, # 0.01 pJ per spike 
            "energy_per_ac": 0.01e-12,    # 0.01 pJ per AC 
            "energy_per_mem_access": 2e-12, # 2 pJ per Byte 
            "gpu_flops": 5e12  # 5 TFLOPS (less relevant)
        }
    },
    
    "hardware_profile": "a100",
    
    # NOTE: this block is NOT wired to anything -- the actual early-exit
    # thresholds live on the model (BranchyLeNet/QDSNN default [0.8, 0.9]) and,
    # for the QDSNN, are learned by the RL agent. These values also used the
    # opposite ordering to the models. Kept only to avoid breaking any external
    # reader; do not treat it as the live policy.
    "early_exit": {
        "threshold_1": 0.8,   # (unused) confidence threshold for first exit
        "threshold_2": 0.9,   # (unused) confidence threshold for second exit
        "use_temperature_scaling": True,
        "temperature_1": 1.0,
        "temperature_2": 1.0
    },
    
    # RL agent settings. Values follow the paper (Sec. III-B):
    # eta=0.1, gamma=0.9, alpha=0.3.
    "rl": {
        "n_confidence_bins": 10,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "exploration_rate": 0.5,
        "exploration_decay": 0.995,
        "min_exploration_rate": 0.01,
        "reward_scale": 0.3,   # weight on the compute-saved bonus in the exit reward
        "episodes": 1000
    },
    
    "data_path": "./data",
    "results_path": "./results",
    "logs_path": "./results/logs",
    "plots_path": "./results/plots",
    
    "models": {
        "lenet": {
            "type": "LeNet",
            "input_size": (1, 28, 28),
            "num_classes": 10
        },
        "snn_lenet": {
            "type": "SNNLeNet",
            "input_size": (1, 28, 28),
            "num_classes": 10,
            "time_steps": STANDARD_SNN_STEPS
        },
        "branchy_lenet": {
            "type": "BranchyLeNet",
            "input_size": (1, 28, 28),
            "num_classes": 10,
            "num_exits": 2
        }
    },
    
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
    
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "lenet_experiments.log"
    }
}