import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time
import cv2
import numpy as np

model_path = './models/face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# For mapping logic 
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405,
                 321, 375, 291, 308, 324, 318, 402, 
                 317, 14, 87, 178, 88, 95, 185, 40,
                 39, 37, 0, 267, 269, 270, 409, 415,
                 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Global variable to store the latest frame with landmarks
latest_annotated_frame = None
last_print_time = 0

# Only projecting mouth
MOUTH_CONNECTIONS = [(MOUTH_INDICES[i], MOUTH_INDICES[i+1]) for i in range(len(MOUTH_INDICES)-1)] + [(MOUTH_INDICES[-1], MOUTH_INDICES[0])]

def draw_mouth_landmarks(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in face_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=MOUTH_CONNECTIONS,  # indices are valid here
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
        )

    return annotated_image

# From google documentation
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])

        # Draw tesselation
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        
        # Draw contours
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        
        # Draw irises
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

def format_detection_results(result: FaceLandmarkerResult, timestamp_ms: int):
    """Format and print face detection results in a readable way"""
    print("\n" + "="*50)
    print(f"TIMESTAMP: {timestamp_ms}ms")
    print(f"FACES DETECTED: {len(result.face_landmarks)}")
    
    if result.face_landmarks:
        for i, face in enumerate(result.face_landmarks):
            print(f"\nFACE {i+1}:")
            print(f"  Total landmarks: {len(face)}")
            print("  Sample landmarks:")
            # Show first 5 landmarks as examples
            for j in range(len(face)):
                landmark = face[j]
                # print(f"    Landmark {j}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
                print("current landmark", landmark)
            
    print("="*50)


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame, last_print_time
    
    # Convert MediaPipe image to numpy array
    rgb_image = output_image.numpy_view()

    # Draw landmarks if faces are detected
    if result.face_landmarks:
        annotated_rgb = draw_mouth_landmarks(rgb_image, result)
        # Convert RGB to BGR for OpenCV display
        latest_annotated_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    else:
        # No faces detected, convert original image to BGR
        latest_annotated_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Print formatted result every 1 second
    current_time = time.time()
    if current_time - last_print_time >= 1.0:
        format_detection_results(result, timestamp_ms)
        last_print_time = current_time   

# Feed options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

def main():
    global latest_annotated_frame
    
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            # Convert BGR to RGB (OpenCV uses BGR, MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calculate timestamp in milliseconds
            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)
            
            # Detect face landmarks asynchronously
            landmarker.detect_async(mp_image, timestamp_ms)
            
            # Display the frame with landmarks (if available) or original frame
            display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame
            cv2.imshow('Face Landmarks', display_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")

if __name__ == "__main__":
    main()
