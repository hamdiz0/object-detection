import os
import pandas as pd
from tqdm import tqdm

# Paths to your dataset
dataset_folder = "./Dataset"
csv_folder = "./csv_folder"

# YOLO folders
train_images = os.path.join(dataset_folder, "train/images")
train_labels = os.path.join(dataset_folder, "train/labels")
test_images = os.path.join(dataset_folder, "test/images")
test_labels = os.path.join(dataset_folder, "test/labels")

# Annotation files from Open Images
train_annotations_file = os.path.join(csv_folder, "train-annotations-bbox.csv")
test_annotations_file = os.path.join(csv_folder, "test-annotations-bbox.csv")
class_descriptions_file = os.path.join(csv_folder, "class-descriptions-boxable.csv")

# Load class descriptions
class_descriptions = pd.read_csv(class_descriptions_file, header=None, names=["class_id", "class_name"])

# Filter for the 'Person' class only
person_class = "Person"
class_map = {row.class_id: 0 for row in class_descriptions.itertuples() if row.class_name == person_class}

if not class_map:
    raise ValueError(f"Class '{person_class}' not found in class descriptions.")

print(f"Using class map: {class_map}")

# Convert Open Images labels to YOLO format
def convert_annotations(annotations_file, labels_folder, images_folder):
    annotations = pd.read_csv(annotations_file)

    # Ensure folders exist
    os.makedirs(labels_folder, exist_ok=True)

    # Iterate over images and their annotations
    for image_id, group in tqdm(annotations.groupby("ImageID")):
        label_path = os.path.join(labels_folder, f"{image_id}.txt")

        # Ensure the corresponding image exists
        image_path = os.path.join(images_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue

        # Write YOLO label file
        with open(label_path, "w") as f:
            for row in group.itertuples():
                if row.LabelName not in class_map:
                    continue

                class_id = class_map[row.LabelName]
                x_center = (row.XMin + row.XMax) / 2
                y_center = (row.YMin + row.YMax) / 2
                width = row.XMax - row.XMin
                height = row.YMax - row.YMin

                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Process train and test sets
print("Processing training set...")
convert_annotations(train_annotations_file, train_labels, train_images)

print("Processing test set...")
convert_annotations(test_annotations_file, test_labels, test_images)

print("Conversion to YOLO format complete!")
