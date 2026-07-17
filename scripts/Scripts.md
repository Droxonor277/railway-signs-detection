# Scripts 
this file shows documentation for all the scripts that are located in the scripts folder. It goes thru individual scripts and their usage.
## Video Frame Extractor - extractor.py

This Python script extracts frames from a video file and saves them as individual images. Frames are grouped and named based on a time-like structure (hours, minutes, seconds, and frame index).

### Requirements
Python 3.x
OpenCV

Install dependencies:

pip install opencv-python


### Usage


> python script.py <video_path> <fps> 



 \<video_path\> — Path to the input video file


\<fps\> — Number of frames per second (used to simulate time tracking)

### Output
A folder named data will be created (if it doesn’t already exist).
Extracted frames are saved as .jpg images inside this folder.
File Naming Format
frame<hour>_<minute>_<second>_<frame>.jpg

Example:

frame0_1_5_12.jpg

This means:

Hour: 0
Minute: 1
Second: 5
Frame index: 12
### How It Works
The script reads the video frame by frame.
Every frame is saved as an image.

## YOLO Annotation Sorter - sort_by_class.py 

This Python script processes YOLO annotation files and sorts them into folders based on their class ID. It only keeps images that contain exactly one annotation, making it useful for creating clean single-class datasets.

### Requirements
Python 3.x


### Usage
Update the script paths:
SOURCE_DIR = "path/to/your/folder"
OUTPUT_DIR = os.path.join(SOURCE_DIR, "sorted_single_class")


Run the script:
> python script.py
### Input Structure

Your source folder should contain matching .jpg and .txt files:

  - image1.jpg
  - image1.txt
  - image2.jpg
  - image2.txt

Each .txt file should follow YOLO format:

<class_id> <x_center> <y_center> <width> <height>
### Output
A new folder will be created:
sorted_single_class/.
Inside, files are grouped by class:
sorted_single_class/
  - class_0/
  - class_1/
  - class_2/

Each folder contains:
The image (.jpg) and its corresponding annotation (.txt)

## YOLO Annotation Collector - sort_empty.py

This script collects YOLO annotation files (`.txt`) and their corresponding images (`.jpg`) from a single folder and copies them into a separate output directory.

It is useful for cleaning or gathering only annotated dataset pairs.


### Requirements

* Python 3.x
* Built-in libraries only (`os`, `shutil`)

###  Usage

1. Update the folder paths in the script:

```
source_folder = os.path.join("C:\\OpenLabeling\\main\\output\\YOLO_darknet")
destination_folder = os.path.join(source_folder, "collected_files")
```

2. Run the script:

```
python script.py
```

###  Input Structure

Your source folder should contain paired files:

  - image1.jpg
  - image1.txt
  - image2.jpg
  - image2.txt

###  Output

A new folder will be created:

```
collected_files/
```

It will contain:

* `.txt` annotation files
* Matching `.jpg` images (if they exist)
