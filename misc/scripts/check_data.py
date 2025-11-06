import h5py
from pathlib import Path
import numpy as np
from PIL import Image  # For saving images

# Define the path to the database
ROOT_DIR = Path(__file__).resolve().parent.parent  # Navigate to the root directory
DATA_DIR = ROOT_DIR / "data"
DATABASE_PATH = DATA_DIR / "thyroid_dataset.h5"

def extract_frames_for_id(database_path, target_id, max_frames=20):
    """Extract up to `max_frames` frames for a specific annot_id and save them as images."""
    if not database_path.exists():
        print(f"❌ Database file not found at {database_path}")
        return

    print(f"📂 Extracting up to {max_frames} frames for annot_id: {target_id}")
    try:
        with h5py.File(database_path, 'r') as db:
            # Open datasets
            annot_ids = db['annot_id']
            images = db['image']

            # Initialize counters and output directory
            extracted_count = 0
            output_dir = DATA_DIR / f"frames_{target_id}"
            output_dir.mkdir(exist_ok=True)

            # Iterate through the dataset to find frames for the target_id
            for i in range(len(annot_ids)):
                # Decode the current annot_id
                current_id = annot_ids[i].decode('utf-8')

                if current_id == target_id:
                    # Save the corresponding image
                    frame = images[i]
                    output_path = output_dir / f"frame_{extracted_count + 1}.png"
                    Image.fromarray(frame).save(output_path)

                    extracted_count += 1
                    if extracted_count >= max_frames:
                        break

            if extracted_count == 0:
                print(f"❌ No frames found for annot_id: {target_id}")
            else:
                print(f"✅ Extracted {extracted_count} frames for annot_id: {target_id}")
                print(f"📂 Frames saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Failed to extract frames: {e}")

# Run the function
extract_frames_for_id(DATABASE_PATH, target_id="130_", max_frames=20)