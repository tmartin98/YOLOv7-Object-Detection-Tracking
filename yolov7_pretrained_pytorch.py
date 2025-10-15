import torch
import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

class YOLOv7Detector:
    def __init__(self, weights_path='yolov7.pt', conf_threshold=0.25, 
                 iou_threshold=0.45, device='cuda'):
        """
        Initialize YOLOv7 detector module
        
        Args:
            weights_path: path to pretrained weights file
            conf_threshold: confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: 'cuda' for GPU or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model from torch hub
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', 
                                     path=weights_path, 
                                     source='github')
        self.model.to(self.device)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.eval()
        
        # COCO class names
        self.class_names = self.model.names
        
        # Tracking storage
        self.track_history = defaultdict(lambda: [])
        self.object_counter = defaultdict(int)
        
    def process_video(self, video_path, output_path=None, 
                     target_classes=None, save_txt=True):
        """
        Process video with detection and tracking
        
        Args:
            video_path: input video path
            output_path: output video path (optional)
            target_classes: list of class IDs to detect (e.g., [0] for person only)
            save_txt: whether to save detections to txt file
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize VideoWriter
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Text file for detections
        txt_path = Path(output_path).stem + '_detections.txt' if save_txt else None
        if txt_path:
            txt_file = open(txt_path, 'w')
        
        frame_count = 0
        total_time = 0
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            start_time = time.time()
            
            # Run detection
            results = self.model(frame)
            
            # Process results
            detections = results.xyxy[0].cpu().numpy()
            
            # Annotate frame
            annotated_frame = self.annotate_frame(
                frame.copy(), 
                detections, 
                frame_count,
                target_classes
            )
            
            # Save detections to txt
            if txt_path:
                self.save_detections_to_txt(
                    txt_file, 
                    frame_count, 
                    detections, 
                    target_classes
                )
            
            # Calculate FPS
            inference_time = time.time() - start_time
            total_time += inference_time
            current_fps = 1 / inference_time
            avg_fps = frame_count / total_time
            
            # Display FPS and frame counter
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Avg: {avg_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display object statistics
            y_offset = 110
            for class_name, count in self.object_counter.items():
                cv2.putText(annotated_frame, f"{class_name}: {count}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 0), 2)
                y_offset += 35
            
            # Save video
            if output_path:
                out.write(annotated_frame)
            
            # Display frame
            cv2.imshow('YOLOv7 Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if txt_path:
            txt_file.close()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total objects detected: {dict(self.object_counter)}")
        
    def annotate_frame(self, frame, detections, frame_id, target_classes=None):
        """
        Annotate frame with bounding boxes and labels
        
        Args:
            frame: input frame
            detections: detection results from model
            frame_id: current frame number
            target_classes: list of class IDs to filter
        
        Returns:
            annotated frame
        """
        # Reset frame object counter
        frame_objects = defaultdict(int)
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            
            # Filter by target classes
            if target_classes and cls_id not in target_classes:
                continue
            
            class_name = self.class_names[cls_id]
            frame_objects[class_name] += 1
            
            # Get color based on class ID
            color = self.get_color_for_class(cls_id)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, 
                         (int(x1), int(y1) - label_h - 10),
                         (int(x1) + label_w, int(y1)), 
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, 
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 255), 2)
            
            # Draw centroid and tracking
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Store track history
            track_key = f"{class_name}_{frame_id % 100}"
            self.track_history[track_key].append((cx, cy))
            
            # Draw track line
            if len(self.track_history[track_key]) > 1:
                points = np.array(self.track_history[track_key][-30:], 
                                 dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
        
        # Update object counter
        for class_name, count in frame_objects.items():
            self.object_counter[class_name] = max(
                self.object_counter[class_name], count
            )
        
        return frame
    
    def save_detections_to_txt(self, file, frame_id, detections, target_classes):
        """
        Save detections to txt file in YOLO format
        
        Args:
            file: opened file object
            frame_id: current frame number
            detections: detection results
            target_classes: list of class IDs to filter
        """
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            
            if target_classes and cls_id not in target_classes:
                continue
            
            # Write in format: frame_id, class_id, x1, y1, x2, y2, confidence
            file.write(f"{frame_id} {cls_id} {x1} {y1} {x2} {y2} {conf}\n")
    
    def get_color_for_class(self, cls_id):
        """
        Generate color based on class ID for consistent visualization
        
        Args:
            cls_id: class ID
            
        Returns:
            BGR color tuple
        """
        np.random.seed(cls_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def detect_image(self, image_path, save_path=None):
        """
        Simple image detection
        
        Args:
            image_path: path to input image
            save_path: path to save annotated image
            
        Returns:
            annotated image and results
        """
        img = cv2.imread(image_path)
        results = self.model(img)
        
        # Render annotated image
        annotated_img = results.render()[0]
        
        if save_path:
            cv2.imwrite(save_path, annotated_img)
        
        return annotated_img, results


# Usage examples
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOv7Detector(
        weights_path='yolov7.pt',
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='cuda'
    )
    
    # Process video - detect all objects
    detector.process_video(
        video_path='input_video.mp4',
        output_path='output_all_objects.mp4',
        save_txt=True
    )
    
    # Process video - detect only persons (class 0 in COCO)
    detector.process_video(
        video_path='input_video.mp4',
        output_path='output_persons_only.mp4',
        target_classes=[0],  # 0 = person in COCO dataset
        save_txt=True
    )
    
    # Detect objects in single image
    annotated, results = detector.detect_image(
        'input_image.jpg',
        save_path='output_image.jpg'
    )
