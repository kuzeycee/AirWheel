import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import BaseOptions, vision

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

CAMERA_INDEX = 0
ENTER_TURN_DEG = 28
EXIT_TURN_DEG = 16
ANGLE_SMOOTHING = 0.3
CALIBRATION_SECONDS = 1.5
MISSING_HANDS_GRACE_FRAMES = 10

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def steering_angle(left_point, right_point):
    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    return math.degrees(math.atan2(dy, dx))


def decide_direction(angle, current_direction):
    if current_direction == "right" and angle >= EXIT_TURN_DEG:
        return "right"
    if current_direction == "left" and angle <= -EXIT_TURN_DEG:
        return "left"
    if angle >= ENTER_TURN_DEG:
        return "right"
    if angle <= -ENTER_TURN_DEG:
        return "left"
    return "straight"


def update_turn_keys(direction, held_keys):
    desired = {direction} if direction in ("left", "right") else set()

    for key in held_keys - desired:
        pyautogui.keyUp(key)
    for key in desired - held_keys:
        pyautogui.keyDown(key)

    return desired


def draw_hand(frame, landmarks, w, h, color):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)
    return pts


def main():
    ensure_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    landmarker = create_landmarker()
    held_keys = set()
    accelerating = False
    direction = "straight"
    smoothed_angle = 0.0
    baseline_angle = 0.0
    calibrating = True
    calibration_start = None
    calibration_samples = []
    missing_frames = 0
    prev_time = time.time()
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            both_hands = False
            angle = 0.0

            if result.hand_landmarks and len(result.hand_landmarks) == 2:
                wrist_points = []
                for landmarks in result.hand_landmarks:
                    pts = draw_hand(frame, landmarks, w, h, (0, 255, 0))
                    wrist_points.append(pts[0])
                wrist_points.sort(key=lambda p: p[0])
                left_point, right_point = wrist_points[0], wrist_points[1]

                both_hands = True
                angle = steering_angle(left_point, right_point)
                cv2.line(frame, left_point, right_point, (0, 255, 255), 3)

            now = time.time()

            if both_hands:
                missing_frames = 0
                smoothed_angle += ANGLE_SMOOTHING * (angle - smoothed_angle)
            else:
                missing_frames += 1

            if calibrating:
                if both_hands:
                    if calibration_start is None:
                        calibration_start = now
                    calibration_samples.append(smoothed_angle)
                    remaining = CALIBRATION_SECONDS - (now - calibration_start)
                    if remaining <= 0:
                        baseline_angle = sum(calibration_samples) / len(calibration_samples)
                        calibrating = False
                    else:
                        cv2.putText(frame, f"Calibrating... hold straight ({remaining:.1f}s)",
                                     (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    calibration_start = None
                    calibration_samples = []
                    cv2.putText(frame, "Raise both hands to calibrate",
                                 (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("Virtual Steering Wheel - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            relative_angle = smoothed_angle - baseline_angle
            hands_lost = missing_frames >= MISSING_HANDS_GRACE_FRAMES

            if both_hands:
                direction = decide_direction(relative_angle, direction)
                held_keys = update_turn_keys(direction, held_keys)
                if not accelerating:
                    pyautogui.keyDown("up")
                    accelerating = True
            elif hands_lost:
                smoothed_angle = 0.0
                relative_angle = 0.0
                direction = "straight"
                held_keys = update_turn_keys(direction, held_keys)
                if accelerating:
                    pyautogui.keyUp("up")
                    accelerating = False
            # else: brief single-frame miss, keep previous direction/keys frozen

            fps = 1 / max(now - prev_time, 1e-6)
            prev_time = now

            label = "NO HANDS" if (not both_hands and hands_lost) else direction.upper()
            cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(frame, f"{relative_angle:.1f} deg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.0f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press C to recalibrate", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Virtual Steering Wheel - press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                baseline_angle = smoothed_angle
    finally:
        for key in held_keys:
            pyautogui.keyUp(key)
        if accelerating:
            pyautogui.keyUp("up")
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
