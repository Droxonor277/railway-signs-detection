# railway-signs-detection

## Project proposal
* [Markdown `assignment-plan.md`](assignment-plan.md)
* [Google Docs](https://docs.google.com/document/d/1Rgy5gC281wqrSPgLawZxtInT__KGmbkHpwaXpPGYZMw/edit?usp=sharing)

## Getting started

Clone the repository and navigate to the project directory:
``` bash
git clone https://gitlab.fel.cvut.cz/marusjur/railway-signs-detection.git
cd railway-signs-detection
```

Create a virtual environment and install dependencies:
``` bash
conda env create -f environment.yml
conda activate railway
```

When you install a new dependency/library via `conda` or `pip`, add it to the `environment.yml` file:
```bash
conda env export --no-builds > environment.yml
```
and commit the changes. The newly added dependency can be installed on other machines (after pulling the changes) using:
```bash
conda env update --file environment.yml --prune
```
## Presentations

Folder [here](https://drive.google.com/drive/u/1/folders/1E0qxdzrGR0fITBUJpXgLQzG6IvLV5vj7)

## Data

Downloaded as mp4 in 1080p resolution, using [yt-dlp](https://github.com/yt-dlp/yt-dlp) tool:

```bash
yt-dlp -f "bestvideo[height<=1080][ext=mp4]" "<url>"
```

Uploaded to [Google Drive - here](https://drive.google.com/drive/folders/1VPJnNZ_3e0sMsH32GSXDa5QXa8EOhSjn?usp=drive_link)

The following videos were downloaded:

- video1: https://www.youtube.com/watch?v=o_fiT9mZbts
- video2: https://www.youtube.com/watch?v=B8xoIp4zKn4
- video3: https://www.youtube.com/watch?v=KaVmBPVYNwI
- video4: https://www.youtube.com/watch?v=h77C6wYXVyI
- video5: https://www.youtube.com/watch?v=kJVu3Z7lbVI
- video6: https://www.youtube.com/watch?v=Z9mvO_eM7do
- video7: https://www.youtube.com/watch?v=UdGtdoRwKIU
- video8: https://www.youtube.com/watch?v=8FrwCZ0PGr8

> all videos were downloaded as `.mp4` in with av01.0.12M.08 codec

### Dataset - Google Drive

On Google Drive in folder [`data`](https://drive.google.com/drive/folders/1EDHr7QXczFPyGCMuiWiOuYT7YQIduT-G?usp=sharing), there are following files/folders:

- folders `video7`, `video6`, `video3`, `video2`, `video1`: annotated video segments images from videos using OpenLabeling in YOLO format. Only a subset: 10 minute segments, frames sampled at 5 fps, handpicked to contain a good variety of signs and signals.  
- `Collected_files_video_1.zip`: labeled frames of shortened video1 using OpenLabeling in YOLO format. All frames annotated. (`video1` above was pruned set of this data) - can be used for testing.
- `dataset.zip`: cut video segments from all videos, organized in folders based on the light signal type or state.


For training the detection, the data used for training the model are described [here](detection/README.md#dataset) - training and validation sets contained all `folders` with video segments, and some clips from `dataset.zip` which were additionally annotated to increase the variety of the data.


## Detection

Trained a custom YOLO11s model on a dataset of railway signs and signals, achieving good performance on the validation set. The detection pipeline processes videos, applies the YOLO model to detect the objects, filters detections based on the significance (temporal filtering of frames with backtracking), optionally classifies detected signs using a separate classifier function, and outputs results in a structured CSV format. The pipeline can also generate annotated videos for visualization and debugging purposes.

> See the dataset, training and pipeline description in [detection/README.md](detection/README.md)


## Notes from meetings

**19/02/2026**

- distance signals have only green and yellow lights (no red !)
- main signals have a red pole with white stripe (sometimes yellow?)
- focus on right track (do not initially analyze double tracks)
- number of light bulbs depend on the signal placement (not important in our case)
- lights can be green, yellow, yellow flashing, red, white, and no light

---

