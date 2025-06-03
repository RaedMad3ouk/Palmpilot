
import cv2
import numpy as np
import pickle
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import pyautogui
import time

# Load model and label names
model = load_model('model/model.keras')
with open('model/gesture_labels.pickle', 'rb') as f:
    label_names = pickle.load(f)

# Mediapipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Sliding window buffer
SEQUENCE_LENGTH = 30
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Start video capture
cap = cv2.VideoCapture(0)
print("[INFO] Starting gesture recognition...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        frame_buffer.append(landmarks)

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            image, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        if len(frame_buffer) == SEQUENCE_LENGTH:
            input_sequence = np.expand_dims(frame_buffer, axis=0)
            prediction = model.predict(input_sequence)[0]
            predicted_idx = np.argmax(prediction)
            predicted_label = label_names[predicted_idx]
            confidence = prediction[predicted_idx] * 100

            # Show result
            cv2.putText(image, f'{predicted_label} ({confidence:.1f}%)',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # OPTIONAL: Simulate actions (e.g., open keyboard)
            if predicted_label == "open_keyboard":
                pyautogui.hotkey('win', 'ctrl', 'o')  # Windows on-screen keyboard
                time.sleep(1)  # prevent re-triggering every frame
            elif predicted_label == "scroll_up":
                pyautogui.scroll(300)
            elif predicted_label == "scroll_down":
                pyautogui.scroll(-300)
            elif predicted_label == "left_click":
                pyautogui.click()
            elif predicted_label == "right_click":
                pyautogui.click(button='right')
            elif predicted_label == "zoom_in":
                pyautogui.keyDown('ctrl')
                pyautogui.scroll(300)
                pyautogui.keyUp('ctrl')
            elif predicted_label == "zoom_out":
                pyautogui.keyDown('ctrl')
                pyautogui.scroll(-300)
                pyautogui.keyUp('ctrl')

    else:
        cv2.putText(image, 'No hand detected', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture Control', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
