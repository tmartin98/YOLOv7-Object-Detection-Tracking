# YOLOv7 Object Detection & Tracking

A comprehensive Python implementation of YOLOv7 for real-time object detection and tracking in videos and images. This project provides an easy-to-use interface for the official YOLOv7 model with additional features like object tracking, filtering, and statistics.

## Features

- 🚀 Real-time object detection using YOLOv7
- 📹 Video processing with frame-by-frame detection
- 🎯 Object tracking with trajectory visualization
- 🔍 Class-specific filtering (e.g., detect only persons)
- 📊 Live FPS counter and object statistics
- 💾 Export detections to text files
- 🖼️ Single image detection support
- ⚡ GPU acceleration support

## Model Information

### Pretrained Weights

This implementation uses YOLOv7 models pretrained on the **MS COCO dataset** (80 classes). The models were trained from scratch without using pretrained weights from other datasets.

**Available Models:**

| Model | Size | AP | FPS | Params | FLOPs |
|-------|------|----|----|--------|-------|
| YOLOv7 | 640 | 51.4% | 161 | 37.6M | 105.2G |
| YOLOv7-X | 640 | 53.1% | 114 | 71.3M | 189.9G |
| YOLOv7-W6 | 1280 | 54.9% | 84 | 70.4M | 360.0G |
| YOLOv7-E6 | 1280 | 56.0% | 56 | 97.2M | 515.2G |
| YOLOv7-D6 | 1280 | 56.6% | 44 | 154.7M | 806.8G |
| YOLOv7-E6E | 1280 | 56.8% | 36 | 151.7M | 843.2G |

### COCO Dataset

The models are trained on the **MS COCO (Microsoft Common Objects in Context)** dataset:

- **Dataset Size:** 330,000 images (118,000 training, 5,000 validation)
- **Number of Classes:** 80 object categories
- **Annotations:** 1.5 million object instances
- **Download:** [https://cocodataset.org](https://cocodataset.org)

**COCO Classes (80 categories):**

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support, optional but recommended)
- Git

### Step 1: Clone Repository

git clone https://github.com/tmartin98/yolov7-detection-tracking.git
cd yolov7-detection-tracking

