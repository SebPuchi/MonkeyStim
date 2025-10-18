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


    def testSingleFrameFaceDetection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened, waiting for initialization...")
        
        # Give camera time to initialize
        time.sleep(2)
        
        # Discard first few frames (often corrupted/black)
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"Discarded frame {i+1} - mean pixel value: {frame.mean():.1f}")
        
        # Now get the actual frame we want to use
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not capture frame")
            exit()

        print(f"Original frame shape: {frame.shape}")

        # Preprocess for face detector (needs 128x128)
        resized_frame = cv2.resize(frame, (128, 128))
        print("resized_frame", resized_frame)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        print("rgb frame", rgb_frame)

        normalized_frame = (rgb_frame.astype(np.float32) / 127.5) - 1.0
        print("nomralized frame", normalized_frame)

        cv2.imwrite("original_camera_frame.jpg", frame)
        print("Saved original_camera_frame.jpg")

        # Add batch dimension
        input_tensor = np.expand_dims(normalized_frame, axis=0)
        print(f"Final input tensor shape: {input_tensor.shape}")
        
        print("Running face detection...")
        self.face_detector.set_tensor(self.face_input_details[0]['index'], input_tensor)
        self.face_detector.invoke()
        
        # Get outputs
        output_0 = self.face_detector.get_tensor(self.face_output_details[0]['index'])  # [1, 896, 16]
        output_1 = self.face_detector.get_tensor(self.face_output_details[1]['index'])  # [1, 896, 1]

        print("OUTPUT 0", output_0[:,0])
        print("OUTPUT 1", output_1[:,0])

        confidence_scores = output_1[0, :, 0]  # Extract all confidence scores
        max_confidence_idx = np.argmax(confidence_scores)  # Find index of highest confidence

        print("HIGHEST CONFIDENCE DETECTION:")
        print(f"Index: {max_confidence_idx}")
        print(f"Confidence: {confidence_scores[max_confidence_idx]:.6f}")
        print(f"OUTPUT 0 (16 values): {output_0[0, max_confidence_idx, :]}")
        print(f"OUTPUT 1 (confidence): {output_1[0, max_confidence_idx, :]}")
        # COnvert back to usable points on image, then draw the box on the image
        # save image with box 
        


def main():
   
    model = FacialDetection()

    model.printModelInfo()
    model.testSingleFrameFaceDetection()



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
