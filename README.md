# YOLOv7 Object Detection & Tracking

Real-time object detection and tracking using YOLOv7 with PyTorch.

## Features

- Real-time object detection with YOLOv7
- Video processing with object tracking
- Class-specific filtering
- Live FPS counter and statistics
- Export detections to text files
- GPU acceleration support

## Installation

git clone https://github.com/yourusername/yolov7-detection.git
cd yolov7-detection
pip install -r requirements.txt

## Download Pretrained Weights

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

## Basic Usage

from yolov7_detector import YOLOv7Detector

Initialize detector
detector = YOLOv7Detector(
weights_path='yolov7.pt',
conf_threshold=0.25,
device='cuda'
)

Process video
detector.process_video(
video_path='input.mp4',
output_path='output.mp4'
)


## Detect Specific Classes

Detect only persons (class 0)
detector.process_video(
video_path='input.mp4',
output_path='output_persons.mp4',
target_classes=
)


## Available Models

| Model | AP | FPS | Download |
|-------|----|----|----------|
| YOLOv7 | 51.4% | 161 | [Link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) |
| YOLOv7-X | 53.1% | 114 | [Link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) |
| YOLOv7-E6E | 56.8% | 36 | [Link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) |

## COCO Dataset Classes

The model is trained on MS COCO dataset with 80 classes:

**Common classes:**
- 0: person
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- 16: dog
- 17: cat

**All classes:** person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Training on Custom Data

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7

python train.py --workers 8 --device 0 --batch-size 32
--data data/custom.yaml --img 640
--weights yolov7_training.pt --name yolov7-custom


## Requirements

- Python 3.8+
- PyTorch >= 1.7.0
- OpenCV >= 4.5.0
- CUDA 11.0+ (optional)

## References

- [YOLOv7 Paper](https://arxiv.org/abs/2207.02696)
- [Official Repository](https://github.com/WongKinYiu/yolov7)
- [COCO Dataset](https://cocodataset.org)

