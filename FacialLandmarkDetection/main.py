import time
import cv2
import numpy as np
import tensorflow as tf

# For mapping logic 
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405,
                 321, 375, 291, 308, 324, 318, 402, 
                 317, 14, 87, 178, 88, 95, 185, 40,
                 39, 37, 0, 267, 269, 270, 409, 415,
                 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Global variable to store the latest frame with landmarks
latest_annotated_frame = None
last_print_time = 0

def checkModels():
    # Step 1: Load both models
    print("Loading face detector model...")
    face_detector = tf.lite.Interpreter("./models/face_detector.tflite")
    face_detector.allocate_tensors()

    print("Loading face landmarks detector model...")
    landmarks_detector = tf.lite.Interpreter("./models/face_landmarks_detector.tflite")
    landmarks_detector.allocate_tensors()

    # Step 2: Inspect both models
    print("\n=== FACE DETECTOR MODEL INFO ===")
    face_input_details = face_detector.get_input_details()
    face_output_details = face_detector.get_output_details()

    print(f"Input details: {len(face_input_details)} input(s)")
    for i, detail in enumerate(face_input_details):
        print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")

    print(f"Output details: {len(face_output_details)} output(s)")
    for i, detail in enumerate(face_output_details):
        print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

    print("\n=== LANDMARKS DETECTOR MODEL INFO ===")
    landmarks_input_details = landmarks_detector.get_input_details()
    landmarks_output_details = landmarks_detector.get_output_details()

    print(f"Input details: {len(landmarks_input_details)} input(s)")
    for i, detail in enumerate(landmarks_input_details):
        print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")

    print(f"Output details: {len(landmarks_output_details)} output(s)")
    for i, detail in enumerate(landmarks_output_details):
        print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

    print("\nBoth models loaded successfully!")

def main():
    global latest_annotated_frame
   
    # Check input and output streams
    checkModels()

    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    start_time = time.time()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        
        # Calculate timestamp in milliseconds
        current_time = time.time()
        timestamp_ms = int((current_time - start_time) * 1000)
        
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")



if __name__ == "__main__":
    main()
