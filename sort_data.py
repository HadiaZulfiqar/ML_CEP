import os
import shutil
import random

# Define the root directory and split ratios
root_dir = "C:/ML_CEP/data"
categories = ["beverages", "snacks"]  # Top-level categories
split_ratios = {"train": 0.9, "val": 0.1, "test": 0}
output_dirs = {key: os.path.join(root_dir, key) for key in split_ratios.keys()}

# Create output directories for train, val, and test
for split, path in output_dirs.items():
    os.makedirs(path, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(path, category), exist_ok=True)

# Split images for each category
for category in categories:
    category_dir = os.path.join(root_dir, category)
    if os.path.isdir(category_dir):
        # Filter only valid image files
        valid_extensions = (".jpg", ".png", ".jpeg")
        images = [f for f in os.listdir(category_dir) if f.lower().endswith(valid_extensions)]
        
        # Debugging: Print detected images
        print(f"Processing category: {category_dir}, Found images: {len(images)}")
        
        if not images:
            print(f"No images found in {category_dir}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        train_idx = int(split_ratios["train"] * len(images))
        val_idx = train_idx + int(split_ratios["val"] * len(images))

        datasets = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:],
        }

        # Copy images to respective folders
        for split, split_images in datasets.items():
            split_dir = os.path.join(output_dirs[split], category)
            for img in split_images:
                src = os.path.join(category_dir, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)
                print(f"Copied: {src} -> {dst}")
