import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np
import torch


class YOLODetector:
    def __init__(self, model_path: str):
        """
        Initialize YOLO detector with model path
        
        Args:
            model_path: Path to the YOLO model weights file
        """
        self.model = YOLO(model_path)
    
    def detect_text_boxes(self, image: np.ndarray) -> list:
        """
        Detect text bounding boxes in the input image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            list: List of text bounding boxes in [x1, y1, x2, y2] format
        """
        
        results = self.model(image)
        results = results[0]
        

        boxes = results.boxes.xyxy
        

        detections = sv.Detections.from_ultralytics(results)
        
        class_names = detections.data['class_name'].tolist()
        text_indices = np.where(np.array(class_names) == 'Text')[0]
        

        text_boxes = []
        for idx in text_indices:
            text_boxes.append(boxes[idx])
        

        boxes_list = [box.round().int().cpu().tolist() for box in text_boxes]
        
        return boxes_list[1:]


if __name__ == "__main__":
    detector = YOLODetector("yolov11x_best.pt")
    image = cv2.imread("sample.jpg")
    text_boxes = detector.detect_text_boxes(image)
    print(f"Detected {len(text_boxes)} text boxes")
