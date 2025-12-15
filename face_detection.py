"""
mtcnn_live_landmarks.py
Live webcam MTCNN face detection + landmark visualization using facenet-pytorch.
"""

import time
import sys

import cv2
import torch
from facenet_pytorch import MTCNN


def main(camera_index=0, device=None):
    # Choose device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create MTCNN detector
    mtcnn = MTCNN(keep_all=True, device=device)

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fps_smooth = None
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from webcam")
                break

            # Convert BGR (OpenCV) to RGB (MTCNN)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect boxes, probabilities, and landmarks
            # boxes: tensor of shape (num_faces, 4)
            # probs: tensor of shape (num_faces,)
            # landmarks: tensor of shape (num_faces, 5, 2)
            boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

            # Draw detections
            if boxes is not None:
                for box, prob, landmark in zip(boxes, probs, landmarks):
                    x1, y1, x2, y2 = map(int, box)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw confidence
                    cv2.putText(frame, f"{prob:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    # landmark is array of 5 (x,y) points: left eye, right eye, nose, left mouth, right mouth
                    for idx, (lx, ly) in enumerate(landmark):
                        lx_i, ly_i = int(lx), int(ly)
                        cv2.circle(frame, (lx_i, ly_i), 3, (0, 0, 255), -1)
                        # optional label for the landmark
                        # names = ['L-eye','R-eye','Nose','L-mouth','R-mouth']
                        # cv2.putText(frame, names[idx], (lx_i+4, ly_i+4), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255),1)

            # Calculate FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)

            # Overlay FPS on frame
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Show
            cv2.imshow("MTCNN Live Landmarks", frame)

            # Quit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # You can pass camera index as first argument, e.g. python mtcnn_live_landmarks.py 1
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(camera_index=cam_idx)
