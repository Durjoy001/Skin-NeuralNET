from scripts.download_data import dataset_dest_path, metadata_dest_path
from src.train_vit import main as train_vit_main

def main():
    # Ensure data is downloaded before training
    if not dataset_dest_path.exists() or not metadata_dest_path.exists():
        print("📥 Downloading required data...")
        import scripts.download_data  # Trigger the download script
    else:
        print("✅ Data already exists. Skipping download.")

    # Start training
    print("🚀 Starting training...")
    train_vit_main()

if __name__ == "__main__":
    main()