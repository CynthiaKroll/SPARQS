import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, QConfig, MinMaxObserver, PerChannelMinMaxObserver, get_default_qat_qconfig, prepare_qat, convert, fuse_modules, fuse_modules_qat
# Import QAT specific layers if needed, or rely on prepare_qat to swap them.
# Forcing QAT types for problematic layers:
# from torch.ao.nn import qat as nnqat 
import snntorch as snn
from snntorch import surrogate
from config import config
import copy # Import needed for the deepcopy check


# Hard bound on the LIF membrane potential (threshold is 1.0). With beta=0.95
# and subtract-reset, a growing per-step input current lets the membrane
# accumulate toward +/-inf over T=32 steps -- exactly the epoch-12 blow-up that
# NaNs the loss, halts training, and poisons the QAT activation observers. A
# clamp well above threshold leaves healthy dynamics (|mem| ~ O(1)) untouched
# while capping the pathological runaway. nan_to_num is a belt-and-suspenders
# guard so a NaN can never thread across timesteps.
MEMBRANE_CLAMP = 8.0


def _clamp_membrane(mem):
    mem = torch.nan_to_num(mem, nan=0.0, posinf=MEMBRANE_CLAMP, neginf=-MEMBRANE_CLAMP)
    return torch.clamp(mem, -MEMBRANE_CLAMP, MEMBRANE_CLAMP)


def get_quant_engine():
    """Pick a quantization engine that this machine actually supports.

    'fbgemm' was hardcoded, but it is not available on every build (this box
    only exposes 'onednn'), which made prepare_qat/convert fail at runtime."""
    supported = list(torch.backends.quantized.supported_engines)
    for preferred in ('fbgemm', 'qnnpack', 'onednn'):
        if preferred in supported:
            return preferred
    return supported[0]


def encode_poisson(x, num_steps, gain=None):
    """Canonical Poisson RATE encoding, shared by the QSNN and the QDSNN and
    matched by the AC/spike/memory counters (same formula as the 5-layer-MLP
    module's utils.encode_poisson_rate).

    Per-timestep spike probability p = clamp(x_norm * gain, 0, 1), where x_norm
    is min-max normalized to [0, 1]. Expected spikes per pixel = T * p, so more
    timesteps give a better rate estimate (standard rate coding).

    Two defects the previous version had:
      - it divided p by num_steps, which makes the expected TOTAL spike count
        <= gain*x regardless of T -- T becomes meaningless and the rate code is
        destroyed (the MLP module documents the same finding);
      - it applied clamp(x, 0, 1) directly to the mean/std-STANDARDIZED CIFAR
        input, which zeroes every below-mean pixel (~half the image). Min-max
        normalizing first keeps the full intensity ordering.
    """
    if gain is None:
        gain = config.get("poisson_gain", 1.0)
    x = x.float()
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    p = torch.clamp(x_norm * gain, 0, 1)
    p = p.unsqueeze(0).expand(num_steps, *x.shape)
    return (torch.rand_like(p) < p).float()


class QuantizedSNNAlexNet(nn.Module):
    """
    Hybrid Quantized SNN AlexNet: Conv/Linear layers quantized to INT8, LIF spiking layers in FP32.
    Uses ONLY main model QuantStub/DeQuantStub. Type transitions handled in forward.
    Uses standard nn.BatchNorm layers, relying on prepare_qat to make them QAT-compatible.
    Supports QAT and threshold recalibration.
    Uses explicit state management for all LIF layers (init_hidden=False).
    """
    def __init__(self, num_classes=10, beta=0.95, thresholds=None, time_steps=None):
        super().__init__()
        # T must match the QDSNN's time window or the power/delay comparison is
        # not like-for-like. Both now default to config["num_steps"].
        if time_steps is None:
            time_steps = config["num_steps"]
        self.beta = beta
        # Thresholds are tricky with re-indexing. Assuming calibration will handle this.
        # For constructor, use a default or a simplified access if needed.
        # For QAT, thresholds are often calibrated later.
        self.time_steps = time_steps
        spike_grad = surrogate.atan()
        
        self.quant = QuantStub() # Overall model entry
        self.dequant = DeQuantStub() # Overall model exit

        # --- Features - WITH intermediate QuantStubs --- 
        self.features_seq = nn.ModuleList([
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),    # 0
            nn.BatchNorm2d(64),                                      # 1
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn1', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 2 -> FP32 out
            nn.MaxPool2d(2, 2),                                      # 3 -> FP32 out
            QuantStub(), # <<<< New 4 (Quantize before next Conv)
            
            # Block 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),            # 5 (old 4)
            nn.BatchNorm2d(192),                                     # 6 (old 5)
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn2', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 7 -> FP32 out
            nn.MaxPool2d(2, 2),                                      # 8 (old 7) -> FP32 out
            QuantStub(), # <<<< New 9 (Quantize before next Conv)

            # Block 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),           # 10 (old 8)
            nn.BatchNorm2d(384),                                     # 11 (old 9)
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn3', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 12 -> FP32 out
            QuantStub(), # <<<< New 13 (Quantize before next Conv)
            
            # Block 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),           # 14 (old 11)
            nn.BatchNorm2d(256),                                     # 15 (old 12)
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn4', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 16 -> FP32 out
            QuantStub(), # <<<< New 17 (Quantize before next Conv)

            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # 18 (old 14)
            nn.BatchNorm2d(256),                                     # 19 (old 15)
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn5', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 20 -> FP32 out
            nn.MaxPool2d(2, 2),                                      # 21 (old 17) -> FP32 out
            QuantStub(), # <<<< New 22 (Quantize before Flatten/Linear)
        ])
        
        self.classifier_seq = nn.ModuleList([
            nn.Flatten(),                                            # 0 (Input comes from features_seq QuantStub)
            
            # Block 6
            nn.Linear(256 * 4 * 4, 4096),                            # 1
            nn.BatchNorm1d(4096),                                    # 2
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn6', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 3 -> FP32 out
            QuantStub(), # <<<< New 4 (Quantize before next Linear)

            # Block 7
            nn.Linear(4096, 4096),                                   # 5 (old 4)
            nn.BatchNorm1d(4096),                                    # 6 (old 5)
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn7', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'), # 7 -> FP32 out
            QuantStub(), # <<<< New 8 (Quantize before final Linear)

            # Output Layer
            nn.Linear(4096, num_classes),                            # 9 (old 7)
        ])

    @staticmethod
    def _to_fp32(x):
        """LIF neurons run in FP32; dequantize the (possibly INT8) activation."""
        if hasattr(x, 'is_quantized') and x.is_quantized:
            x = x.dequantize()
        if x.dtype != torch.float32:
            x = x.float()
        return x

    def _run_seq(self, x, seq, mems):
        """Run one timestep through a ModuleList, threading LIF membrane state."""
        snn_idx = 0
        for layer in seq:
            if isinstance(layer, snn.Leaky):
                x = self._to_fp32(x)
                mem = mems[snn_idx]
                if mem is None or mem.shape != x.shape:
                    mem = torch.zeros_like(x)
                x, mem = layer(x, mem)
                mems[snn_idx] = _clamp_membrane(mem)  # cap the T=32 runaway
                snn_idx += 1
            else:
                x = layer(x)
        return x

    def forward(self, x):
        """x: [Time, Batch, C, H, W] spike train, or [Batch, C, H, W] to encode.

        Returns the per-timestep output stack [Time, Batch, num_classes]."""
        if x.dim() == 4:  # raw images -> Poisson spike train (same encoder as QDSNN)
            x = encode_poisson(x, self.time_steps)

        feature_mems = [None] * sum(1 for m in self.features_seq if isinstance(m, snn.Leaky))
        classifier_mems = [None] * sum(1 for m in self.classifier_seq if isinstance(m, snn.Leaky))

        out_rec = []
        for t in range(self.time_steps):
            xt = self.quant(x[t])
            xt = self._run_seq(xt, self.features_seq, feature_mems)
            xt = self._run_seq(xt, self.classifier_seq, classifier_mems)
            out_rec.append(xt)

        return self.dequant(torch.stack(out_rec, dim=0))

# --- Advanced QAT QConfig ---
def get_advanced_qat_qconfig():
    return QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8, 
            qscheme=torch.per_tensor_affine,
            # quant_min=0,  # Default for MinMaxObserver for quint8
            # quant_max=255 # Default for MinMaxObserver for quint8
        ),
        weight=PerChannelMinMaxObserver.with_args( # Changed to PerChannel for weights as it's common
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            # quant_min=-128, # Default for PerChannelMinMaxObserver for qint8
            # quant_max=127  # Default for PerChannelMinMaxObserver for qint8
        )
    )

# --- QAT Preparation (Update exclusion logic) ---
def prepare_quantized_snn_alexnet_for_qat(model):
    model.train() # Ensure model is in training mode

    torch.backends.quantized.engine = get_quant_engine()
    print(f"Using quantization engine: {torch.backends.quantized.engine}")

    model.qconfig = get_advanced_qat_qconfig()
    print(f"Set top-level model.qconfig: activation obs: {type(model.qconfig.activation())}, weight obs: {type(model.qconfig.weight())}")

    print("--- Applying QConfig to submodules and excluding SNNs (before FP32 fusion) ---")
    for name, module in model.named_modules():
        if isinstance(module, snn.Leaky):
            module.qconfig = None
            print(f"  Excluded SNN layer: {name} ({type(module)}) by setting qconfig=None.")
        elif isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, QuantStub, DeQuantStub)):
            if not hasattr(module, 'qconfig') or module.qconfig is None:
                module.qconfig = model.qconfig
                print(f"  Assigned model qconfig to: {name} ({type(module)}). Qconfig: {module.qconfig}")
            else:
                print(f"  QConfig already present or explicitly set for: {name} ({type(module)}). Current: {module.qconfig}")
        elif name == "features_seq" or name == "classifier_seq" or name == "": # Skip parent modules for qconfig print
            pass
        # else:
            # print(f"  Module {name} ({type(module)}) will not be explicitly assigned a qconfig, may inherit or be ignored by QAT.")

    print("--- Attempting FP32 Fusion (Conv-BN, Linear-BN) --- BEFORE prepare_qat --- UPDATING INDICES --- ")
    fp32_modules_to_fuse_str_pairs = []
    # Need to use NEW indices after adding QuantStubs
    # Conv(0)-BN(1), Conv(5)-BN(6), Conv(10)-BN(11), Conv(14)-BN(15), Conv(18)-BN(19)
    feature_conv_indices = [0, 5, 10, 14, 18]
    for i in feature_conv_indices:
        if isinstance(model.features_seq[i], nn.Conv2d) and isinstance(model.features_seq[i+1], nn.BatchNorm2d):
            fp32_modules_to_fuse_str_pairs.append([f'features_seq.{i}', f'features_seq.{i+1}'])
    # Linear(1)-BN(2), Linear(5)-BN(6)
    classifier_linear_indices = [1, 5] # New indices in classifier_seq
    for i in classifier_linear_indices:
         if i+1 < len(model.classifier_seq) and isinstance(model.classifier_seq[i], nn.Linear) and isinstance(model.classifier_seq[i+1], nn.BatchNorm1d):
            fp32_modules_to_fuse_str_pairs.append([f'classifier_seq.{i}', f'classifier_seq.{i+1}'])

    if fp32_modules_to_fuse_str_pairs:
        print(f"  Identified FP32 pairs for fusion (using new indices): {fp32_modules_to_fuse_str_pairs}")
        # TRAIN mode: fuse_modules_qat keeps BatchNorm LIVE via intrinsic
        # ConvBn2d/LinearBn1d modules (folded only at convert). Fusing in eval
        # mode BEFORE training (as this used to do) folds the freshly-initialized
        # BN -- an identity -- into the conv/linear weights and deletes it, so
        # the network trains with no normalization at all.
        if model.training:
            fuse_modules_qat(model, fp32_modules_to_fuse_str_pairs, inplace=True)
        else:
            fuse_modules(model, fp32_modules_to_fuse_str_pairs, inplace=True)
        print("  FP32 module fusion attempt completed.")

        print("--- Verifying FP32 Fusion (checking if BNs are gone) ---")
        bn_layers_after_fp32_fusion = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name.endswith('.bn'):
                    # Live BN inside an intrinsic ConvBn/LinearBn module created
                    # by train-mode fusion (fuse_modules_qat). EXPECTED: it keeps
                    # normalizing during QAT and convert() folds it at the end.
                    continue
                # Check if this BN was part of a fusion pair
                was_targeted = any(name == pair[1] for pair in fp32_modules_to_fuse_str_pairs)
                if was_targeted:
                    print(f"  ERROR: BatchNorm '{name}' ({type(module)}) was targeted for FP32 fusion but STILL EXISTS.")
                    bn_layers_after_fp32_fusion.append(name)
                else:
                    print(f"  INFO: BatchNorm '{name}' ({type(module)}) exists but was not targeted for FP32 fusion.")
        if not any(name in dict(model.named_modules()) for pair in fp32_modules_to_fuse_str_pairs for name in pair if isinstance(dict(model.named_modules()).get(name), (nn.BatchNorm2d, nn.BatchNorm1d)) and any(name == p[1] for p in fp32_modules_to_fuse_str_pairs)):
            print("  SUCCESS: All targeted BatchNorms appear to be fused (no longer exist as separate BN modules).")
        else:
            print(f"  WARNING: Some targeted BatchNorms may still exist: {bn_layers_after_fp32_fusion}. Check logs above.")
    else:
        print("  No FP32 Conv-BN or Linear-BN pairs identified for fusion.")

    print("--- Calling prepare_qat (after FP32 fusion attempt) ---")
    prepare_qat(model, inplace=True)
    print("--- Model after prepare_qat call ---")

    print("--- Post-prepare_qat Diagnostics ---")
    qat_conv_count = 0
    qat_linear_count = 0
    bn_still_present_after_prepare_qat = 0
    problematic_bn1d_obs = False

    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.qat.Conv2d):
            qat_conv_count += 1
            # print(f"  QAT Conv: {name} ({type(module)}), QConfig: {module.qconfig}")
        elif isinstance(module, torch.ao.nn.qat.Linear):
            qat_linear_count += 1
            # print(f"  QAT Linear: {name} ({type(module)}), QConfig: {module.qconfig}")
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if name.endswith('.bn'):
                # Live BN inside an intrinsic QAT module (nniqat.ConvBn2d /
                # LinearBn1d) -- expected with train-mode fusion; folded at convert.
                continue
            bn_still_present_after_prepare_qat +=1
            print(f"  CRITICAL WARNING: BatchNorm layer '{name}' ({type(module)}) still exists after FP32 fusion AND prepare_qat. This is unexpected.")
            if isinstance(module, nn.BatchNorm1d) and (not hasattr(module, 'activation_post_process') or module.activation_post_process is None):
                print(f"    Additionally, this BatchNorm1d ('{name}') does NOT have an observer. This will cause issues.")
                problematic_bn1d_obs = True
            elif hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                print(f"    This BatchNorm ('{name}') has an observer: {type(module.activation_post_process)}. QConfig: {module.qconfig}")
        elif isinstance(module, QuantStub):
            if not hasattr(module, 'activation_post_process') or module.activation_post_process is None:
                 print(f"  WARNING: QuantStub '{name}' does not have an observer after prepare_qat.")
            # else: print(f"  QuantStub '{name}' has observer: {type(module.activation_post_process)}")
        elif isinstance(module, DeQuantStub):
            if not hasattr(module, 'activation_post_process') or module.activation_post_process is None:
                 print(f"  WARNING: DeQuantStub '{name}' does not have an observer after prepare_qat.")
            # else: print(f"  DeQuantStub '{name}' has observer: {type(module.activation_post_process)}")

    print(f"  Found {qat_conv_count} QAT Conv layers and {qat_linear_count} QAT Linear layers after prepare_qat.")
    if bn_still_present_after_prepare_qat == 0:
        print("  SUCCESS: No standalone BatchNorm layers remain after FP32 fusion and prepare_qat.")
    else:
        print(f"  ERROR: {bn_still_present_after_prepare_qat} BatchNorm layer(s) still present after FP32 fusion and prepare_qat.")
    if problematic_bn1d_obs:
        print("  ERROR: At least one BatchNorm1d layer failed to get an observer, even if it (incorrectly) still exists.")

    print("--- End of prepare_quantized_snn_alexnet_for_qat ---")
    return model

# --- Threshold Calibration Utility (Modified for new structure if needed) ---
def calibrate_snn(model, calib_loader, device, time_steps=None, num_batches=20):
    if time_steps is None:
        time_steps = config["num_steps"]
    # This function might need adjustment if it relies on specific module names/paths
    # that have changed due to QuantStub/DeQuantStub insertions.
    # For now, keeping it as is, assuming it finds SNN layers by type.
    model.eval()
    snn_layers_to_calibrate = []
    for name, module in model.named_modules():
        if isinstance(module, snn.Leaky): # Check type directly
             snn_layers_to_calibrate.append((name, module))

    for name, module in snn_layers_to_calibrate:
        if hasattr(module, 'reset_mechanism'): # Ensure it's an SNNTorch neuron with state
            # The original calibrate_snn logic for collecting activations via model(spike_train)
            # and then accessing module.mem might be problematic if .mem is not directly populated
            # or if the forward pass for calibration doesn't match training.
            # A simpler approach for SNNTorch layers with init_hidden=False:
            # We need to capture the membrane potential *before* spiking, over time.
            # This is complex. The original code relied on module.mem, let's see if it still works.
            # A more robust way would be to use hooks to capture inputs to the activation function of SNN.
            print(f"Attempting to calibrate SNN layer: {name}")
            
            # Simplified placeholder for threshold calibration logic:
            # The original logic in the attached file `quantized_snn_alexnet.py` for calibration
            # iterates through `model.named_modules()` and checks `if hasattr(module, 'mem')`.
            # This should still find the `snn.Leaky` layers.
            # The key is whether `module.mem` gets populated correctly during the calibration forward pass.
            # Given `init_hidden=False`, `mem` is passed explicitly.
            # The provided calibration logic seems to fetch `module.mem` *after* a full model forward pass.
            # This means `module.mem` would be the state at the *end* of `time_steps`.
            # This might be okay for percentile-based calibration.
            
            # Re-using a simplified version of the original calibration logic:
            reservoir_size = 10000
            reservoir = []
            count = 0
            # Temporarily store original device of module.threshold if it's a tensor
            threshold_device = 'cpu'
            if isinstance(module.threshold, torch.Tensor):
                threshold_device = module.threshold.device

            with torch.no_grad():
                for i, (images, _) in enumerate(calib_loader):
                    if i >= num_batches: break
                    x_calib = images.to(device)
                    spike_input_calib = encode_poisson(x_calib, time_steps)
                    _ = model(spike_input_calib) 

                    if hasattr(module, 'mem') and module.mem is not None:
                        mems = module.mem.detach().cpu().flatten() 
                        for val in mems:
                            count += 1
                            if len(reservoir) < reservoir_size:
                                reservoir.append(val)
                            else:
                                j = torch.randint(0, count, (1,)).item()
                                if j < reservoir_size:
                                    reservoir[j] = val
                    else:
                        pass # Or handle error

            if reservoir:
                all_acts = torch.tensor(reservoir)
                if len(all_acts) > 0:
                    new_thr = torch.quantile(all_acts, 0.95).item()
                    # Ensure threshold is a scalar float, then convert to tensor if layer expects tensor
                    if isinstance(module.threshold, torch.Tensor):
                         module.threshold = torch.tensor(new_thr, device=threshold_device) 
                    else: # Assuming it's a float attribute
                         module.threshold = new_thr 
                    print(f"Calibrated {name} threshold to {new_thr:.4f} (using {len(reservoir)} samples)", flush=True)
                else:
                    print(f"No membrane potentials collected for {name}. Threshold not calibrated.")
            else:
                print(f"Reservoir empty for {name}. Threshold not calibrated.")


# --- Conversion Example --- (Updated to use reconstruction)
def convert_quantized_snn_alexnet(model_qat_trained): # model_qat_trained is the model AFTER QAT training
    print("--- Entering convert_quantized_snn_alexnet (Reconstruction Strategy) ---")
    
    # Get necessary __init__ args from the trained QAT model
    num_classes_arg = -1
    if hasattr(model_qat_trained, 'classifier_seq') and len(model_qat_trained.classifier_seq) > 0 and hasattr(model_qat_trained.classifier_seq[-1], 'out_features'):
        num_classes_arg = model_qat_trained.classifier_seq[-1].out_features
    else:
        # Fallback or error if num_classes cannot be inferred
        print("ERROR: Could not infer num_classes from model_qat_trained. Defaulting to 10, but please check.")
        num_classes_arg = 10 # Or raise an error

    beta_arg = model_qat_trained.beta if hasattr(model_qat_trained, 'beta') else 0.95 # Default if not found
    time_steps_arg = model_qat_trained.time_steps if hasattr(model_qat_trained, 'time_steps') else 4 # Default

    print(f"Reconstructing a clean FP32 model shell for conversion with: num_classes={num_classes_arg}, beta={beta_arg}, time_steps={time_steps_arg}")

    # 1. Create a new, fresh FP32 shell instance of your base model
    model_fp32_shell = QuantizedSNNAlexNet(
        num_classes=num_classes_arg,
        beta=beta_arg,
        time_steps=time_steps_arg
        # Thresholds are not passed here; they are part of SNN Leaky module state or handled by calibration on the QAT model if needed
    )
    model_fp32_shell.train() # Start in train mode for preparation steps

    # 2. Prepare this shell for QAT exactly like you did for the original model
    print("--- Preparing the FP32 shell for QAT (to get QAT module structure and observers) ---")
    # Ensure prepare_quantized_snn_alexnet_for_qat sets up qconfigs, fusions, and calls prepare_qat
    model_qat_shell = prepare_quantized_snn_alexnet_for_qat(model_fp32_shell) 
    print("--- FP32 shell preparation for QAT finished. Shell now has QAT modules and observers. ---")

    print("--- Copying state_dict from trained QAT model to QAT shell ---")
    try:
        # It's crucial that model_qat_trained and model_qat_shell have compatible state_dicts.
        # This means the names and types of layers (post-fusion, post-prepare_qat) must align.
        # Differences in SNN layer internal state (if not part of state_dict) are bypassed.
        model_qat_shell.load_state_dict(model_qat_trained.state_dict())
        print("State_dict loaded successfully from trained QAT model into QAT shell.")
    except RuntimeError as e_sd:
        print(f"ERROR loading state_dict into QAT shell: {e_sd}")
        print("This might indicate a mismatch in module names or structure.")
        print("Keys in trained QAT model (source):", list(model_qat_trained.state_dict().keys()))
        print("Keys in QAT shell (destination):", list(model_qat_shell.state_dict().keys()))
        # For more detailed diff:
        # trained_keys = set(model_qat_trained.state_dict().keys())
        # shell_keys = set(model_qat_shell.state_dict().keys())
        # print("Keys only in trained model:", trained_keys - shell_keys)
        # print("Keys only in shell model:", shell_keys - trained_keys)
        raise
    except Exception as e_other_sd:
        print(f"UNEXPECTED error loading state_dict: {e_other_sd}")
        raise

    # 3. Now, prepare this populated QAT shell for actual conversion
    print("--- Preparing QAT shell for ACTUAL conversion (eval, cpu) ---")
    model_qat_shell.eval()
    model_qat_shell.cpu()
    torch.backends.quantized.engine = get_quant_engine()  # Ensure engine is set

    print("--- Calling torch.ao.quantization.convert --- ON QAT SHELL (populated with trained data) --- ")
    model_quantized = None
    try:
        model_quantized = torch.ao.quantization.convert(model_qat_shell, inplace=False) # inplace=False is safer
        print("--- torch.ao.quantization.convert SUCCEEDED --- ON QAT SHELL --- ")
    except RuntimeError as e_convert_runtime:
        print(f"--- torch.ao.quantization.convert FAILED (RuntimeError). Error: {e_convert_runtime} --- ON QAT SHELL --- ")
        print("This is the point where the original deepcopy error would have occurred on the full model_qat_trained.")
        print("If it happens here, the issue might be with convert's internal handling of the QAT shell itself, even after state_dict load.")
        raise
    except Exception as e_convert_other:
        print(f"--- torch.ao.quantization.convert FAILED (Other Exception). Error: {e_convert_other} --- ON QAT SHELL --- ")
        raise

    print("--- Model structure AFTER conversion (from QAT shell) --- FINAL QUANTIZED MODEL --- ")
    for name, module in model_quantized.named_modules():
         if isinstance(module, (torch.ao.nn.quantized.Conv2d, torch.ao.nn.quantized.Linear,
                                torch.ao.nn.quantized.Quantize, torch.ao.nn.quantized.DeQuantize,
                                nn.Identity, nn.MaxPool2d, nn.Flatten, snn.Leaky # snn.Leaky should still be here
                                )) or not list(module.children()): # Print leaf modules too
            print(f"  Post-Convert: {name} - {type(module)}")
            # Check scale/zp for quantized layers
            if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
                print(f"    Quant Params: scale={module.scale}, zero_point={module.zero_point}")

    print("--- Exiting convert_quantized_snn_alexnet --- RETURNING CONVERTED MODEL --- ")
    return model_quantized

# --- Usage Example ---
# from models.quantized_snn_alexnet import QuantizedSNNAlexNet, prepare_quantized_snn_alexnet_for_qat, convert_quantized_snn_alexnet
# model = QuantizedSNNAlexNet(num_classes=10, beta=0.95, thresholds=None, time_steps=4)
# model = prepare_quantized_snn_alexnet_for_qat(model)
# # ... QAT training loop ...
# model = convert_quantized_snn_alexnet(model)
