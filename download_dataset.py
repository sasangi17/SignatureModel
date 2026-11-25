from huggingface_hub import snapshot_download

print("Starting signature dataset download...")

dataset_path = snapshot_download(
    repo_id="T0KII/signatures-dataset",
    repo_type="dataset",
    local_dir="raw_dataset"
)

print("\nDataset downloaded successfully!")
print("Dataset location:", dataset_path)
