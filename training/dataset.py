import kagglehub
from pathlib import Path

required_path = Path("./data/raw") # run it while you are in the training folder

path = kagglehub.dataset_download("mahmoudreda55/satellite-image-classification", output_dir=str(required_path))

print("Path to dataset files:", path)