# Railway signs detection from locomotive dashcam footage

## Description
Events at the end of 2025 on Czech and Slovak railways show that human error continues to play a significant role in railway safety. At least three incidents caused by unauthorized passage of a stop signal resulted in several serious injuries and dozens of minor injuries (Jablonov na Turňov on October 13, 2025, Pezinok on November 9, 2025, and Zliv on November 20, 2025). The aim of this project is to explore the possibilities of automatic detection of light railway signals from a camera located on a railway vehicle. In the future, this detection could serve as part of an assistance system for train drivers on tracks where the implementation of the ETCS system is not yet in sight. Our task is to design and test an algorithm for detecting signals in images and also their signal characters (Stop, Warning, Clear, Expect speed..., Speed... and clear/warning). We plan to use Matlab Image Processing Toolbox, Python and OpenCV library, or any other framework to solve the task. We will Evaluate the detection success of the proposed algorithm, if possible, on video recordings with different lighting conditions (day, night, fading light, driving into the sun) Our priority will be for detection in sunny day clear lighting. We will Use videos published on YouTube as our data source, see Literature Description.

The objective of this project is to design, implement, and evaluate a computer vision system in Python that:

- Detects railway light signals in cab view videos  
- Classifies their signal aspects  
- Detects a specific blinking yellow signal  
- Evaluates detection and classification performance under varying lighting conditions  

The system will be trained and tested using publicly available cab view videos.

---

## 2. Acceptance Criteria

1. **Video Playback Application**
   - The application must support common video formats (e.g., mp4, avi, mkv).
   - The detection algorithm shall run in the background during playback.
   - Detected signals must be visualized using rectangular bounding boxes.

2. **Signal Detection**
   - Detect main, distant, and static railway light signals in cab-view videos.
   - Detection must happen when the signal is in specified area in the video/windshield. for maximum resolution and better detection


3. **Detection Performance**
   - Recall > 80 %
   - Precision > 80 %
   - Measured on a manually annotated test dataset.

4. **Signal Classification**
   - Classification accuracy > 80 % (evaluated on correctly detected signals).

5. **Demonstration Requirement**
   - The final presentation video must show at least 20 detected signals.

---

## 3. Work Tasks

These tasks can later be assigned either to individual team members or subdivided further among team members.

### Task 1: Data Collection and Annotation

**Description:** Preparation of a structured dataset from publicly available cab-view videos.
**Estimated Workload:** 30–40 hours
**Basic Requirements:** Laptop, internet connection, sufficient storage space

#### Subtasks

#### 1.1 Video Selection `(done)`
- Select representative cab-view videos (day, night, sunset, glare).
- Ensure diversity in environment and signal appearance.

**Recommended Tools:**
- `yt-dlp` (open-source downloader)

**Deliverable:**
- List of selected videos


#### 1.2 Manual Annotation
- Select a tool suitalbe for frame annotation and labeling.
- Draw bounding boxes around visible signals.
- Assign labels to signal aspects.
- Label blinking yellow signal sequences.

**Recommended Open-Source Tools:**
- CVAT, Label Studio, OpenLabeling

**Deliverable:**
- Annotated dataset
- Annotation files in [YOLO format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format) (or compatible with chosen detection framework)


### Task 2: Signal Detection (Object Detection)

**Assignee: Juraj Marusic**
**Description:** Development of an algorithm to detect railway main and distant signals in video frames.
**Estimated Workload:** 40–50 hours
**Basic Requirements:** Laptop (GPU preferred for training)

#### Subtasks

#### 2.1 Data Preparation
- Convert annotated dataset to [YOLO format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
- Split dataset into training, validation, and test sets
- Image resizing and normalization
- Data augmentation (brightness, contrast, blur, noise) -increase the training dataset size (optional, could be used as a future improvement if the initial model performance is poor)

**Deliverable:**
- Preprocessed training dataset


#### 2.2 Model Selection
Evaluate and compare object detection architectures.
Candidate Models: YOLOv8, YOLOv11 (faster)

**Deliverable:**
- Selected model with justification


#### 2.3 Model Training and Optimization
- Train detection model
- Hyperparameter tuning
- Validation testing

**Deliverable:**
- Trained detection model
- Performance graphs


#### 2.4 Detection Performance Evaluation
- Precision, recall, F1-score calculation
- Precision–recall curves
- Error analysis

**Deliverable:**
- Graphs and summary tables


### Task 3: Signal Classification (Aspect Recognition)

**Description:** Classification of detected signals based on light configuration and color.
**Estimated Workload:** 30–40 hours
**Basic Requirements:** Laptop

#### Subtasks

#### 3.1 Data Preparation
- Crop detected signal regions
- Experiment with image enhancement techniques:
  - HSV color space conversion
  - Saturation adjustment
  - Histogram equalization

**Deliverable:**
- Preprocessed cropped dataset


#### 3.2 Classification Method Selection

Possible approaches:

- Convolutional Neural Network (custom small CNN)
- Transfer learning (e.g., MobileNet)
- Classical computer vision:
  - HSV color thresholding
  - Blob detection
  - Hough Circle Transform

**Deliverable:**
- Selected classification approach


#### 3.3 Training and Evaluation
- Train classifier
- Compute confusion matrix
- Evaluate classification accuracy

**Deliverable:**
- Confusion matrix
- Accuracy graphs


### Task 4: Blinking Yellow Signal Detection

**Description:** Detection of temporal blinking behavior of a specific yellow signal aspect.
**Estimated Workload:** 20–30 hours
**Basic Requirements:** Laptop

#### Subtasks

#### 4.1 Temporal Analysis
- Track detected yellow signal across consecutive frames
- Analyze brightness variation over time
- Identify periodic intensity changes

Possible methods:
- Frame differencing
- Intensity curve analysis
- Short temporal window signal tracking

**Deliverable:**
- Signal intensity graphs


#### 4.2 Blinking Classification Logic
- Define blinking frequency threshold
- Distinguish steady yellow from blinking yellow

**Deliverable:**
- Blinking detection module
- Evaluation summary


### Task 5: Video Player Implementation

**Description:** Development of a video playback tool integrating detection and classification.
**Estimated Workload:** 25–30 hours
**Basic Requirements:** Laptop

#### Subtasks

#### 5.1 Basic Player Implementation
- Load video
- Play/pause functionality
- Frame-by-frame processing

**Tools:**
- OpenCV, Optional: PyQt for GUI

**Deliverable:**
- Functional video player


#### 5.2 Visualization classification results
- Draw bounding boxes of detected signals or display them in a side panel
- Display classification labels

**Deliverable:**
- Integrated visualization module

---

## 4. Milestones

| Milestone | Date | Description |
|------------|------|------------|
| Dataset Prepared | 20 March 2026 | Videos selected, frames extracted, at least 50% annotated |
| Detection Prototype | 10 April 2026 | Initial detection model trained and evaluated |
| Classification Prototype | 25 April 2026 | Classification implemented and evaluated |
| Blinking Detection Implemented | 5 May 2026 | Temporal logic integrated |
| Integrated System Ready | 15 May 2026 | End-to-end system functional |
| Final Testing & Presentation | 22 May 2026 | Final evaluation complete, presentation video ready |

---

## Gantt project plan

> TODO: use this <https://www.onlinegantt.com/#/gantt> or this <https://sourceforge.net/projects/ganttproject/>

---

## 5. Resources

### Hardware
- Laptops (16 GB RAM recommended)
- Optional NVIDIA GPU
- at least 30 GB storage


---

## 6. Risk Assessment

| Risk Description | Severity | Likelihood | Project Phase | Mitigation Strategy |
|------------------|----------|------------|---------------|--------------------|
| Insufficient annotated data | High | Low | Data Collection | Increase dataset size early |
| Poor detection accuracy | High | Medium | Detection | Use data augmentation and alternative models |
| Lighting variability causes failures | High | High | Detection & Classification | Use data augmentation |
| Blinking detection unreliable | Medium | Medium | Blinking Task | Use longer temporal window |
| Hardware performance limitations | Medium | Medium | Detection | Use smaller models or cloud GPU |
| Workload imbalance within team | Medium | Medium | All phases | Early task allocation and monitoring |
| Low video quality | Medium | Low | Data Collection | Select high-resolution source videos |


## Literature

1. SŽ D1 ČÁST PRVNÍ - Dopravní a návěstní předpis pro tratě nevybavené evropským vlakovým zabezpečovačem, část III. NÁVĚSTIDLA A NÁVĚSTI, https://provoz.spravazeleznic.cz/portal/Show.aspx?oid=2271537
2. video https://www.youtube.com/watch?v=o_fiT9mZbts
3. video https://www.youtube.com/watch?v=B8xoIp4zKn4
4. video https://www.youtube.com/watch?v=KaVmBPVYNwI
5. video https://www.youtube.com/watch?v=h77C6wYXVyI
6. video https://www.youtube.com/watch?v=kJVu3Z7lbVI
