import os 
import shutil
import random
from sklearn.model_selection import train_test_split

# Define the paths to images and labels
data_dir = "D:\FKIE\git_workspace\SyntheticDataGeneration\dataset_city_park"
images_dir = os.path.join(data_dir, "images")
labels_dir = os.path.join(data_dir, "masks")

all_images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('jpg', 'jpeg'))])
all_labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.png')])

# Ensure the lists are correctly paired
assert len(all_images) == len(all_labels), "Mismatch between images and labels count."

# Pair images and labels to keep them together during the split
all_pairs = list(zip(all_images, all_labels))

# Step 1: Split into train+val and test sets (80% train+val, 20% test)

# train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, shuffle=True)
# shuffle=True

# Step 2: Further split train+val into train and validation sets (90% train, 10% validation)
train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.1, shuffle=False)

# Summary of split sizes
print(f"Total Images: {len(all_images)}")
print(f"Training Images: {len(train_pairs)}, Validation Images: {len(val_pairs)}, Test Images: {len(test_pairs)}")

# Optional: Verify no overlap between sets (for debugging)
train_images, train_labels = zip(*train_pairs)
val_images, val_labels = zip(*val_pairs)
test_images, test_labels = zip(*test_pairs)
assert len(set(train_images) & set(test_images)) == 0, "Train and Test sets overlap!"
assert len(set(train_images) & set(val_images)) == 0, "Train and Validation sets overlap!"
assert len(set(val_images) & set(test_images)) == 0, "Validation and Test sets overlap!"

# Directory structure for saving splits
output_dir = "custom_city_park"
split_dirs = {
    "train": {"images": os.path.join(output_dir, "train", "Images"), "labels": os.path.join(output_dir, "train", "Labels")},
    "val": {"images": os.path.join(output_dir, "val", "Images"), "labels": os.path.join(output_dir, "val", "Labels")},
    "test": {"images": os.path.join(output_dir, "test", "Images"), "labels": os.path.join(output_dir, "test", "Labels")}
}

# Create directories for each split
for split, paths in split_dirs.items():
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)

# Function to copy files to their respective split directories
def save_split(pairs, split):
    for img_path, label_path in pairs:
        shutil.copy(img_path, split_dirs[split]["images"])
        shutil.copy(label_path, split_dirs[split]["labels"])

# Save each split
save_split(train_pairs, "train")
save_split(val_pairs, "val")
save_split(test_pairs, "test")

print("Train, validation, and test images/labels saved to disk successfully.")
