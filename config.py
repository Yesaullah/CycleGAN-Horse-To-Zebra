from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "horse2zebra"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

DEVICE = "cuda"

image_size = 256
batch_size = 4
lr = 0.0002
lambda_cycle = 10
lambda_identity = 5
n_epochs = 200
decay_epoch = 100
num_residual_blocks = 9

num_workers = 8
betas = (0.5, 0.999)

