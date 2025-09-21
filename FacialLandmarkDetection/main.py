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

class FacialDetection:

    def __init__(self):
        print("Loading face detector model...")

        self.face_detector = tf.lite.Interpreter("./models/face_detector.tflite")
        self.face_detector.allocate_tensors()

        print("Loading face landmarks detector model...")

        self.landmarks_detector = tf.lite.Interpreter("./models/face_landmarks_detector.tflite")
        self.landmarks_detector.allocate_tensors()

        self.face_input_details = self.face_detector.get_input_details()
        self.face_output_details = self.face_detector.get_output_details()


        print("MODELS LOADED SUCCESSFULLY")

    def printModelInfo(self):
        print("\n=== FACE DETECTOR MODEL INFO ===")

        print(f"Input details: {len(self.face_input_details)} input(s)")
        for i, detail in enumerate(self.face_input_details):
            print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")

        print(f"Output details: {len(self.face_output_details)} output(s)")
        for i, detail in enumerate(self.face_output_details):
            print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

        print("\n=== LANDMARKS DETECTOR MODEL INFO ===")
        landmarks_input_details = self.landmarks_detector.get_input_details()
        landmarks_output_details = self.landmarks_detector.get_output_details()

        print(f"Input details: {len(landmarks_input_details)} input(s)")
        for i, detail in enumerate(landmarks_input_details):
            print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")

        print(f"Output details: {len(landmarks_output_details)} output(s)")
        for i, detail in enumerate(landmarks_output_details):
            print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")


    def testSingleFrame(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not capture frame")
            exit()

        print(f"Original frame shape: {frame.shape}")

        # Preprocess for face detector (needs 128x128)
        resized_frame = cv2.resize(frame, (128, 128))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)

        print(f"Input tensor shape: {input_tensor.shape}")


def main():
   
    model = FacialDetection()

    model.printModelInfo()
    model.testSingleFrame()


    # cap = cv2.VideoCapture(0)
    # 
    # # Check if camera opened successfully
    # if not cap.isOpened():
    #     print("Error: Could not open camera")
    #     return
    # 
    # print("Camera opened successfully!")
    # start_time = time.time()
    # 
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     
    #     if not ret:
    #         print("Can't receive frame. Exiting ...")
    #         break
    #     
    #     # Calculate timestamp in milliseconds
    #     current_time = time.time()
    #     timestamp_ms = int((current_time - start_time) * 1000)
    #     
    #     # Display the frame
    #     cv2.imshow('Camera Feed', frame)
    #     
    # # When everything is done, release the capture and close windows
    # cap.release()
    # cv2.destroyAllWindows()
    # print("Camera released and windows closed")



if __name__ == "__main__":
    main()
