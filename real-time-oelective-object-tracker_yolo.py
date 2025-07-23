import cv2
import numpy as np
import os

class YOLOTracker:
    def __init__(self, weights_path="yolov3.weights", config_path="yolov3.cfg", 
                 names_path="coco.names", confidence_threshold=0.6, nms_threshold=0.4):
        """
        Initialize YOLO Tracker with configurable parameters
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.MAX_LOST_FRAMES = 25
        
        # Load class names
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Classes file not found: {names_path}")
        
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load YOLO network
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError("YOLO weights or config file not found")
        
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        
        # Initialize tracking variables
        self.tracker = None
        self.initBB = None
        self.lost_frames = 0
        self.tracking_active = False
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def detect_objects(self, frame):
        """
        Detect objects using YOLO
        """
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
        # Process detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Calculate bounding box coordinates
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_width = int(detection[2] * w)
                    box_height = int(detection[3] * h)
                    x = int(center_x - box_width / 2)
                    y = int(center_y - box_height / 2)
                    
                    boxes.append([x, y, box_width, box_height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        if boxes:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 
                                     self.confidence_threshold, self.nms_threshold)
            return boxes, confidences, class_ids, indexes
        
        return [], [], [], []
    
    def draw_detections(self, frame, boxes, confidences, class_ids, indexes):
        """
        Draw detection boxes and labels on frame
        """
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with background
                label_text = f"{label} {int(confidence * 100)}%"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), color, -1)
                cv2.putText(frame, label_text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def start_tracking(self, frame):
        """
        Start manual object selection for tracking
        """
        self.initBB = cv2.selectROI("Object Detection & Tracking", frame, 
                                   fromCenter=False, showCrosshair=True)
        if self.initBB[2] > 0 and self.initBB[3] > 0:  # Valid selection
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, self.initBB)
            self.tracking_active = True
            self.lost_frames = 0
            return True
        return False
    
    def update_tracking(self, frame):
        """
        Update object tracking
        """
        if not self.tracking_active or self.tracker is None:
            return False, None
        
        success, box = self.tracker.update(frame)
        
        if success:
            self.lost_frames = 0
            return True, box
        else:
            self.lost_frames += 1
            if self.lost_frames >= self.MAX_LOST_FRAMES:
                self.reset_tracking()
            return False, None
    
    def reset_tracking(self):
        """
        Reset tracking state
        """
        self.tracker = None
        self.initBB = None
        self.lost_frames = 0
        self.tracking_active = False
    
    def draw_tracking(self, frame, box, success):
        """
        Draw tracking box and status
        """
        if success and box is not None:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "Tracking Active", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Tracking Lost ({self.lost_frames}/{self.MAX_LOST_FRAMES})", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def draw_instructions(self, frame):
        """
        Draw usage instructions on frame
        """
        instructions = [
            "Press 's' to select object for tracking",
            "Press 'r' to reset tracking",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 60 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Main execution loop
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting YOLO Detection + CSRT Tracking System")
        print("Controls:")
        print("  's' - Select object for tracking")
        print("  'r' - Reset tracking")
        print("  'q' - Quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Add instructions to frame
                frame = self.draw_instructions(frame)
                
                if not self.tracking_active:
                    # Detection mode
                    boxes, confidences, class_ids, indexes = self.detect_objects(frame)
                    frame = self.draw_detections(frame, boxes, confidences, class_ids, indexes)
                    
                    # Show detection count
                    detection_count = len(indexes) if len(indexes) > 0 else 0
                    cv2.putText(frame, f"Objects detected: {detection_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                else:
                    # Tracking mode
                    success, box = self.update_tracking(frame)
                    frame = self.draw_tracking(frame, box, success)
                
                cv2.imshow("Object Detection & Tracking", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and not self.tracking_active:
                    if self.start_tracking(frame):
                        print("Tracking started successfully")
                    else:
                        print("Invalid selection, please try again")
                elif key == ord('r'):
                    self.reset_tracking()
                    print("Tracking reset")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """
    Main function to run the tracker
    """
    try:
        # Initialize and run tracker
        tracker = YOLOTracker(
            weights_path="yolov3.weights",
            config_path="yolov3.cfg", 
            names_path="coco.names",
            confidence_threshold=0.6,
            nms_threshold=0.4
        )
        tracker.run(camera_id=0)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure YOLO files are in the current directory:")
        print("- yolov3.weights")
        print("- yolov3.cfg") 
        print("- coco.names")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()