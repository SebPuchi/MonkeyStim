import numpy as np
import torch
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import threading
import time

# model
from blazeface import BlazeFace

def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    plt.show()


# Shared frames and timestamps
frame_left, frame_right = None, None
ts_left, ts_right = 0.0, 0.0
lock = threading.Lock()
running = True

def capture(cam, side):
    """Capture frames from one camera in a separate thread."""
    global frame_left, frame_right, ts_left, ts_right, running

    while running:
        ret, frame = cam.read()
        if not ret:
            print(f"[{side}] Can't receive frame. Exiting thread...")
            break

        timestamp = time.time()  # seconds since epoch (high precision)

        with lock:  # ensure safe updates
            if side == 'left':
                frame_left = frame
                ts_left = timestamp
            else:
                frame_right = frame
                ts_right = timestamp

    cam.release()

def main():
    print("PyTorch version:", torch.__version__)

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    front_net = BlazeFace().to(gpu)
    front_net.load_weights("blazeface.pth")
    front_net.load_anchors("anchors.npy")

    # back_net = BlazeFace(back_model=True).to(gpu)
    # back_net.load_weights("blazefaceback.pth")
    # back_net.load_anchors("anchorsback.npy")

    # thresholds
    front_net.min_score_thresh = 0.75
    front_net.min_suppression_threshold = 0.3


    global running

    left = cv.VideoCapture(0); left.set(cv.CAP_PROP_BUFFERSIZE, 1)
    right = cv.VideoCapture(1); right.set(cv.CAP_PROP_BUFFERSIZE, 1)


    if not right.isOpened() or not left.isOpened():
        print("Cannot open camera")
        return

    # Start parallel capture threads
    t_right = threading.Thread(target=capture, args=(right, 'right'), daemon=True)
    t_left = threading.Thread(target=capture, args=(left, 'left'), daemon=True)
    t_right.start()
    t_left.start()

    while True:
        with lock:
            if frame_left is not None and frame_right is not None:
                # Compute time difference
                dt = abs(ts_left - ts_right)
                    

                f_left = cv.cvtColor(frame_left, cv.COLOR_BGR2RGB)
                f_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)

                img_batch = np.vstack((np.expand_dims(cv.resize(f_left, (128,128)),0), np.expand_dims(cv.resize(f_right, (128,128)),0)))
                print(img_batch.shape)
               
                front_detections = front_net.predict_on_batch(img_batch)
                print("LEFT", front_detections[0])
                print("RIGHT", front_detections[1])

                # Combine frames horizontally
                # combined = cv.hconcat([f_left, f_right])
                # cv.imshow('stereo', combined)
                
                plot_detections(f_left, front_detections[0])
                plot_detections(f_right, front_detections[1])
                break

                # Optionally display timing info
                print(f"Timestamp diff: {dt * 1000:.2f} ms")

        if cv.waitKey(1) == ord('q'):
            running = False
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
