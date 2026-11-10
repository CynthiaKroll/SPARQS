import torch

from config import config
from data_loader import get_data_loaders
from evaluate_quantized_pair import load_qsnn, run_eval

CHECKPOINT = "outputs/alexnet_qsnn_20epochs_lr5e5_qat.pth"
TIME_STEPS = 32
BATCH_SIZE = 16

config["batch_size"] = BATCH_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader, _ = get_data_loaders(
    distributed=False,
    augment=False
)

model = load_qsnn(
    CHECKPOINT,
    TIME_STEPS,
    device
)

result = run_eval(
    model=model,
    name="QSNN QAT checkpoint",
    test_loader=test_loader,
    device=device,
    time_steps=TIME_STEPS,
    is_branchy=False
)

print("\nQSNN QAT evaluation completed")
print(f"Device: {device}")
print(f"Test accuracy: {result['accuracy_pct']:.2f}%")
print(f"Average latency per sample: {result['latency_s_per_sample'] * 1000:.3f} ms")
