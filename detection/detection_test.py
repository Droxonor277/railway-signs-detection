import os
import torch
from ultralytics import YOLO  

IMGSZ = 1440 # 1280
# file = '../videos/video2/video2-Zvolen-Kosice-14-02-10-00.mp4'
# file = '../../videos/video2/video2-Zvolen-Kosice-16-02-30-00.mp4'
# file = '../../videos/video6/video6-Zilina-Bratislava-4-00-30-00.mp4'
file = 'detection-test.mp4'

# file = '../videos/video2-6-segments.mp4'
# file = '../test.mp4'
save_file = filename = os.path.splitext(os.path.basename(file))[0]
save_path = '/media/jur0/SharedOS/ING/04/PTVY/railway-signs-detection/detection/results/n6/tmp'

# model = YOLO('models/yolo11s_n3.pt')  # Nano for speed  
model = YOLO('models/yolo11s_n6.pt')  # Nano for speed  

results = model(
    source=file,
    project=save_path,
    name=filename,
    save=True, 
    conf=0.5,
    stream=True,      # Process frames one-by-one, not load entire video
    imgsz=IMGSZ,        # Resize frames (lower if video is high-res) changeed from 640
    device= 'cuda:0',  # 'cuda:0' or 'cpu'; CPU often leaks less initially
    verbose=True,     # Reduce logging overhead
    # batch=4,     # Process one frame at a time to minimize memory usage
)
for r in results:
    torch.cuda.empty_cache()  # Clear GPU memory each frame