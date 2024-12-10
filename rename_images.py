import os

# Define the root directory
root_dir = "C:/ML_CEP/data/snacks"

# Function to rename images in each subdirectory
def rename_images_in_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for i, image in enumerate(images, start=1):
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, f"image{i}.jpg")
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Process each subdirectory
for sub_dir in os.listdir(root_dir):
    sub_dir_path = os.path.join(root_dir, sub_dir)
    if os.path.isdir(sub_dir_path):  # Ensure it's a directory
        print(f"Renaming images in: {sub_dir_path}")
        rename_images_in_folder(sub_dir_path)
