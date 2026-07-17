# This script sorts YOLO anotations based on their class as prescribed in class_list.txt
import os
import shutil

SOURCE_DIR = "path/to/your/folder" #FIXME
OUTPUT_DIR = os.path.join(SOURCE_DIR, "sorted_single_class")#FIXME

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(SOURCE_DIR):
    if not file.endswith(".txt"):
        continue

    txt_path = os.path.join(SOURCE_DIR, file)
    base_name = os.path.splitext(file)[0]
    jpg_path = os.path.join(SOURCE_DIR, base_name + ".jpg")

    if not os.path.exists(jpg_path):
        print(f"Missing image for: {file}")
        continue

    # Read all non-empty lines
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Keep only files with exactly ONE annotation
    if len(lines) != 1:
        print(f"Skipped (invalid annotation count): {file}")
        continue

    parts = lines[0].split()
    if len(parts) < 1:
        print(f"Skipped (invalid format): {file}")
        continue

    class_id = parts[0]

    # Create class folder
    class_folder = os.path.join(OUTPUT_DIR, f"class_{class_id}")
    os.makedirs(class_folder, exist_ok=True)

    # Copy files
    shutil.copy2(jpg_path, os.path.join(class_folder, base_name + ".jpg"))
    shutil.copy2(txt_path, os.path.join(class_folder, base_name + ".txt"))

print("Sorting completed.")