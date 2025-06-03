# ✋Palmpilot

##  Introduction
This project is the final-year capstone for the Machine Learning specialization at Holberton School.

It is an upgraded system that **uses hand gestures to control a computer**, including:
- Media actions like scroll and zoom
- Direct mouse interaction like move, click, drag

The system combines:
- Computer Vision (MediaPipe)
- Sequence Modeling (LSTM Neural Networks)
- GUI Automation (PyAutoGUI)

With no external APIs, it runs entirely on the local machine using just a webcam.

---

## 🏗️ Project Architecture

| Component                  | Description                                                                            |
|----------------------------|----------------------------------------------------------------------------------------|
| **Gesture Data Collection** | Captures hand landmark sequences using MediaPipe, saved as `.npy` files.               |
| **Sequence Classifier**     | LSTM-based neural network trained to classify gesture sequences over time.             |
| **Real-Time Prediction**    | Feeds live landmark sequences from the webcam into the model for gesture recognition.  |
| **Function Mode**           | Maps gestures to actions (scroll, zoom, switch to mouse).                              |
| **Mouse Mode**              | Maps hand movement to cursor, with left/right clicks, double-click, and drag support.  |

---



##  Features

- Control mouse cursor using index fingertip
- Perform left/right clicks, double clicks, and drag using finger pinches
- Switch between **gesture mode** and **mouse mode** using thumbs-up gesture
- Smooth, real-time predictions based on 60-frame landmark sequences
- Confidence thresholds + cooldown timers to reduce false triggers

---

##  Data Collection

- Captured custom dataset using webcam
- Saved 60-frame sequences (21 landmarks × 3D) per gesture
- Collected under varied lighting and conditions for robustness
- Final gesture classes:
  - `zoom in`
  - `zoom out`
  - `scroll up`
  - `scroll down`
  - `switch` (thumbs-up)
  - `neutral` (no gesture)
- Stored as `.npy` files and combined into a `sequence_data.pickle` file for training

---

##  Model

- **Architecture**:
  - LSTM (64) → Dropout → LSTM (64) → Dropout → Dense (64, ReLU) → Dense (softmax)
- **Loss**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Training**: 35 epochs, batch size 16, ~96% test accuracy
- **Output**: `model.keras` file for live prediction

---

##  Gesture-to-Action Mapping



---

##  Dependencies

- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow / Keras
- PyAutoGUI
- NumPy

---


## Conclusion
This project showcases how machine learning + computer vision can power intuitive, touchless control systems.
By combining LSTM gesture recognition with real-time OS interaction, we bridge cutting-edge AI and practical user applications — creating a system that’s both innovative and accessible.



