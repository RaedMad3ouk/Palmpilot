import os
import cv2
import numpy as np
import mediapipe as mp

# Config
BASE_DIR = './data/sequences'
SEQUENCE_LENGTH = 60

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

cap = cv2.VideoCapture(0)

gesture_label = input("Enter gesture name (e.g., scroll_up): ").strip().lower()
gesture_dir = os.path.join(BASE_DIR, gesture_label)

if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

sequence_id = len(os.listdir(gesture_dir))
print(f"Saving to: {gesture_dir}")
print("Press 's' to start recording a sequence.")
print("Press 'ESC' to exit.")

while True:
    sequence = []
    frame_count = 0

    while frame_count < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        landmarks = []
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sequence.append(landmarks)
            frame_count += 1

            # Draw hand
            mp.solutions.drawing_utils.draw_landmarks(
                image, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, f'Recording [{frame_count}/{SEQUENCE_LENGTH}]',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No Hand Detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Collecting Sequence', image)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Save sequence
    if len(sequence) == SEQUENCE_LENGTH:
        np.save(os.path.join(gesture_dir, f'{gesture_label}_seq_{sequence_id}.npy'),
                np.array(sequence))
        print(f"Saved {gesture_label}_seq_{sequence_id}.npy")
        sequence_id += 1

    print("Press 's' to record again, or 'ESC' to stop.")
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()