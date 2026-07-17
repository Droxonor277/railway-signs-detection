"""
Crop bounding box regions from raw YOLO-annotated video folders, organized by class.

The raw dataset structure expected under <raw_path>:
    <raw_path>/<video_dir>/imgs/*.jpg
    <raw_path>/<video_dir>/yolo/*.txt   (YOLO format: class_id cx cy w h)

For each annotated bounding box, the corresponding image region is cropped and
saved into:
    <output_dir>/<class_name>/<image_stem>_<idx>.jpg

Usage:
    python crop_by_class.py <raw_path> <output_dir> <class_list>

Example:
    python crop_by_class.py data/raw data/crops data/class_list.txt
"""

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop bounding boxes from raw YOLO-annotated data, organized by class."
    )
    parser.add_argument("raw_path", type=Path, help="path to the raw data root (contains video subdirs)")
    parser.add_argument("output_dir", type=Path, help="path to the output directory")
    parser.add_argument("class_list", type=Path, help="path to class_list.txt (one class name per line)")
    return parser.parse_args()


def load_class_names(class_list_path: Path) -> list[str]:
    with open(class_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def yolo_bbox_to_pixel(cx: float, cy: float, w: float, h: float,
                        img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """convert normalized yolo bbox to pixel coordinates (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def process_video_dir(
    video_dir: Path,
    class_names: list[str],
    output_dir: Path,
) -> int:
    """process all images in a video dir and save crops. returns number of crops saved."""
    images_dir = video_dir / "imgs"
    labels_dir = video_dir / "yolo"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"  skipping '{video_dir.name}': imgs or yolo folder not found")
        return 0

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    crops_saved = 0

    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  warning: could not read image {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"  warning: unexpected label format in {label_path}: '{line}'")
                continue

            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            if class_id >= len(class_names):
                print(f"  warning: class id {class_id} out of range in {label_path}")
                continue

            x1, y1, x2, y2 = yolo_bbox_to_pixel(cx, cy, w, h, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                print(f"  warning: degenerate bbox in {label_path}, line {idx}")
                continue

            crop = img[y1:y2, x1:x2]
            class_name = class_names[class_id]
            out_path = output_dir / class_name / f"{img_path.stem}_{idx}.jpg"
            cv2.imwrite(str(out_path), crop)
            crops_saved += 1

    return crops_saved


def main() -> None:
    args = parse_args()

    if not args.class_list.exists():
        raise FileNotFoundError(f"class list not found: {args.class_list}")

    class_names = load_class_names(args.class_list)
    print(f"classes: {class_names}")

    # create output subdirs for each class upfront
    for class_name in class_names:
        (args.output_dir / class_name).mkdir(parents=True, exist_ok=True)

    # each subdirectory in raw_path is a video folder
    video_dirs = sorted(d for d in args.raw_path.iterdir() if d.is_dir())
    if not video_dirs:
        raise RuntimeError(f"no subdirectories found in {args.raw_path}")

    total_crops = 0
    for video_dir in video_dirs:
        print(f"processing: {video_dir.name}")
        n = process_video_dir(video_dir, class_names, args.output_dir)
        print(f"  saved {n} crops")
        total_crops += n

    print(f"\ndone - total crops saved: {total_crops}")


if __name__ == "__main__":
    main()
