#this script moves anotated picures and their anotations into separeate file,
#it will need its inputs in one folder

import os
import shutil

# paths (edit these)
source_folder = os.path.join("C:\OpenLabeling\main\output\YOLO_darknet")#FIXME
destination_folder = os.path.join(source_folder, "collected_files")#FIXME

# create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.lower().endswith(".txt"):
        txt_path = os.path.join(source_folder, filename)

        # skip empty .txt files
        if os.path.getsize(txt_path) == 0:
            print(f"Skipped empty file: {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        jpg_name = base_name + ".jpg"
        jpg_path = os.path.join(source_folder, jpg_name)

        # copy .txt file
        shutil.copy2(txt_path, destination_folder)
        print(f"Copied: {filename}")

        # if matching .jpg exists, copy it too
        if os.path.exists(jpg_path):
            shutil.copy2(jpg_path, destination_folder)
            print(f"Copied: {jpg_name}")
        else:
            print(f"No matching JPG for: {filename}")

print("Done.")