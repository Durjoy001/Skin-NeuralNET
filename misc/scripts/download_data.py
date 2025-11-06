import urllib.request
from pathlib import Path
from tqdm import tqdm

# Define destination path relative to the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent  # Navigate to the root directory
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URL and destination
dataset_url = "http://berrypidrive.duckdns.org:8000/api/public/dl/vtTkh8rl/dataset.hdf5"
dataset_dest_path = DATA_DIR / "thyroid_dataset.h5"

# Metadata URL and destination
metadata_url = "http://berrypidrive.duckdns.org:8000/api/public/dl/vtTkh8rl/metadata.csv"
metadata_dest_path = DATA_DIR / "metadata.csv"

def download_with_progress(url, dest_path):
    """Download a file with a progress bar."""
    response = urllib.request.urlopen(url)
    total_size = int(response.getheader('Content-Length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name)

    with open(dest_path, 'wb') as file:
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            file.write(buffer)
            progress_bar.update(len(buffer))
    progress_bar.close()

try:
    # Check if dataset already exists
    if dataset_dest_path.exists():
        print(f"✅ Dataset already exists at {dataset_dest_path.resolve()}")
    else:
        # Download dataset with progress
        print(f"📦 Downloading dataset from {dataset_url} ...")
        download_with_progress(dataset_url, dataset_dest_path)
        print(f"✅ Download complete! Saved to {dataset_dest_path.resolve()}")

    # Check if metadata already exists
    if metadata_dest_path.exists():
        print(f"✅ Metadata already exists at {metadata_dest_path.resolve()}")
    else:
        # Download metadata with progress
        print(f"📦 Downloading metadata from {metadata_url} ...")
        download_with_progress(metadata_url, metadata_dest_path)
        print(f"✅ Download complete! Saved to {metadata_dest_path.resolve()}")

except Exception as e:
    print(f"❌ Failed to download files: {e}")