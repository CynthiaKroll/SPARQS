"""
Configuration settings for the AlexNet project.
"""

# --- Theoretical energy constants (single source of truth) ---
# The SPARQ paper (Eq. 4 and Eq. 5) specifies E_mem = 80e-12 J/byte
# (Horowitz 2014), the same value the 5-layer-MLP module uses. This module
# previously used 4 pJ/byte, so its energy numbers were 20x off the paper's
# methodology (and inconsistent with the MLP results).
E_MEM_J_PER_BYTE_PAPER = 80e-12     # 80 pJ/Byte - value used by the SPARQ paper
E_MEM_J_PER_BYTE_DDR3_45NM = 4e-12  # 4 pJ/Byte - DDR3 45nm (kept for reference only)

E_MAC_32BIT_J = 4.6e-12  # Energy for a 32-bit floating-point Multiply-Accumulate
E_MAC_8BIT_J = 0.2e-12   # Energy for an 8-bit integer Multiply-Accumulate
E_AC_8BIT_J = 0.03e-12   # Energy for an 8-bit integer Accumulate (Add) - typical for SNNs

E_MEM_J_PER_BYTE = E_MEM_J_PER_BYTE_PAPER

E_AC_NEUROMORPHIC_J = 0.01e-12         # 0.01 pJ per AC - more realistic for neuromorphic
E_MEM_NEUROMORPHIC_J_PER_BYTE = 2e-12  # 2 pJ per Byte - more realistic neuromorphic memory

config = {
    # Data settings
    "data_dir": "data",
    "input_size": (3, 32, 32),  # CIFAR-10 image size
    "num_classes": 10,
    
    # Training settings
    "batch_size": 128,
    "test_batch_size": 128,
    "learning_rate": 0.001,  # unified across all three SPARQ modules
    "weight_decay": 1e-4,
    # The paper states NO epoch count anywhere; 20 is the documented number,
    # unified across all three modules (MLP / LeNet / AlexNet).
    "num_epochs": 20,
    # Mixup data augmentation (paper Sec. III-C). alpha=0 disables it; 0.4
    # matches the MLP module. Overridable via train.py --mixup_alpha.
    "mixup_alpha": 0.4,
    "early_stopping_patience": 10,
    
    # SNN settings
    # num_steps is the SINGLE time window used by BOTH the QSNN and the QDSNN.
    # Previously the QSNN trainer hardcoded T=4 while the QDSNN used T=32, which
    # made the power/delay comparison meaningless.
    "num_steps": 32,
    "beta": 0.95,     # LIF membrane decay, unified with MLP/LeNet (was 0.85)
    # Gain for Poisson rate encoding: P(spike per step) = clamp(x_norm*gain, 0, 1),
    # x_norm min-max normalized. gain=1.0 => firing probability == normalized
    # intensity (no saturation); with gain=2.0 every pixel above 0.5 fires on
    # EVERY timestep. Matches the MLP module's canonical encoder. Do NOT divide
    # by num_steps -- that makes total spikes T-independent and destroys the
    # rate code.
    "poisson_gain": 1.0,
    "thresholds": [0.4, 0.8],  # Default early-exit confidence thresholds

    # RL agent settings. Values follow the paper (Sec. III-B):
    # eta=0.1, gamma=0.9, alpha=0.3.
    "rl": {
        "agent_type": "q_learning",
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "exploration_rate": 0.5,
        "exploration_decay": 0.995,
        "min_exploration_rate": 0.01,
        "n_confidence_bins": 10,
        # Weight on the "compute saved by exiting early" term of the exit reward.
        "reward_scale": 0.3,
    },

    # Quantization settings
    "qat_learning_rate": 5e-05,  # (unused fallback; the trainers take --lr)
    "qat_num_epochs": 20,        # unified with num_epochs / other modules
    "ptq_calibration_batches": 100,  # Increased from 20
    
    # Hardware constants for different profiles
    "hardware_profile": "nvidia_a100",  # Default hardware profile
    "hardware_constants": {
        "default": {
            "energy_per_mac": E_MAC_32BIT_J,
            "energy_per_ac": E_AC_8BIT_J,
            "energy_per_mem_access": E_MEM_J_PER_BYTE,
            "gpu_flops": 10e12  # 10 TFLOPS
        },
        "nvidia_a100": {
            "energy_per_mac": E_MAC_32BIT_J,
            "energy_per_ac": E_AC_8BIT_J,
            "energy_per_mem_access": E_MEM_J_PER_BYTE,
            "gpu_flops": 19.5e12  # 19.5 TFLOPS
        },
        "neuromorphic": {
            "energy_per_mac": E_MAC_8BIT_J,
            "energy_per_ac": E_AC_NEUROMORPHIC_J,
            "energy_per_mem_access": E_MEM_NEUROMORPHIC_J_PER_BYTE,
            "gpu_flops": 5e12  # 5 TFLOPS (less relevant)
        }
    },
    "E_AC_PAPER_J": 0.1e-12,  # 0.1 pJ, Energy per AC from DDECS paper (TABLE I, INT ADD/ACC)
    "E_AC_NEUROMORPHIC_J": E_AC_NEUROMORPHIC_J,
    "E_MEM_NEUROMORPHIC_J_PER_BYTE": E_MEM_NEUROMORPHIC_J_PER_BYTE,

    # Training SNN configurations
    "snn_train_batch_size": 128
}

# --- Per-operation constants for 45nm CMOS (reference only) ---
# NOTE: this dict is deliberately NOT assigned back into
# config["hardware_constants"]. Doing so used to CLOBBER the hardware profiles
# above (default / nvidia_a100 / neuromorphic), so every lookup of
# config["hardware_constants"]["nvidia_a100"] silently fell through to a
# hardcoded default.
E_INT_ADD_45NM_J = 0.1e-12  # Integer ADD operation
E_INT_MUL_45NM_J = 3.1e-12  # Integer MUL operation
E_INT_MAC_45NM_J = 3.2e-12  # Integer MAC (MUL + ADD) operation

E_FP_ADD_45NM_J = 0.9e-12   # Floating Point ADD operation
E_FP_MUL_45NM_J = 3.7e-12   # Floating Point MUL operation
E_FP_MAC_45NM_J = 4.6e-12   # Floating Point MAC (MUL + ADD) operation

cmos_45nm_constants = {
    "E_INT_ADD_J": E_INT_ADD_45NM_J,
    "E_INT_MUL_J": E_INT_MUL_45NM_J,
    "E_INT_MAC_J": E_INT_MAC_45NM_J,
    "E_FP_ADD_J": E_FP_ADD_45NM_J,
    "E_FP_MUL_J": E_FP_MUL_45NM_J,
    "E_FP_MAC_J": E_FP_MAC_45NM_J,
    "E_MEM_J_PER_BYTE": E_MEM_J_PER_BYTE,
    "E_AC_SNN_PAPER_J": 0.1e-12,  # paper's SNN AC cost
}
