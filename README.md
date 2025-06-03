# Palmpilot
Introduction
This project is the final-year capstone for the Machine Learning specialization at Holberton School.
It is an upgraded system that uses hand gestures to control a computer — including media actions (scroll, zoom) and direct mouse interaction (move, click, drag).

The system combines:
 Computer Vision (MediaPipe)
 Sequence Modeling (LSTM Neural Networks)
 GUI Automation (PyAutoGUI)

With no external APIs needed, the system runs entirely on the local machine using just a webcam.

Project Architecture
Component	Description
Gesture Data Collection	Captures hand landmark sequences using MediaPipe and saves them as .npy files.
Sequence Classifier	An LSTM-based neural network trained to classify gesture sequences over time.
Real-Time Prediction	Feeds live landmark sequences from the webcam into the model for gesture recognition.
Function Mode	Maps gestures to actions like scroll up/down, zoom in/out, or switching to mouse mode.
Mouse Mode	Maps hand movement to cursor control, supports left/right clicks, double-clicks, and drag.

Features
Control mouse cursor with index fingertip

Perform left/right clicks, double clicks, and drag using finger pinches

Switch between gesture control and mouse control modes using a thumbs-up gesture

Smooth, real-time predictions based on 60-frame landmark sequences

Confidence thresholding and cooldown timers to prevent false triggers

Data Collection
Captured own dataset using webcam

Saved 60-frame sequences of hand landmarks (21 points × 3D)

Gestures include:

Zoom In

Zoom Out

Scroll Up

Scroll Down

Switch (Thumbs-Up)

Neutral (No Gesture)

Stored raw .npy files per sequence and combined into a single .pickle file for training

Model
Architecture:

Two stacked LSTM layers (64 units)

Dropout layers for regularization

Dense softmax output layer for multi-class classification

Loss Function: Categorical Cross-Entropy

Optimizer: Adam

Training: 35 epochs, batch size 16, ~96% test accuracy

Final Output: model.keras saved and loaded for real-time use

Gesture-to-Action Mapping
Gesture	Action
Scroll Up	pyautogui.scroll(+300)
Scroll Down	pyautogui.scroll(-300)
Zoom In	pyautogui.hotkey('ctrl', '+')
Zoom Out	pyautogui.hotkey('ctrl', '-')
Switch	Switch between gesture and mouse modes
Mouse Mode	Move cursor, right click, left click, drag

Dependencies
Python 3.x

OpenCV

TensorFlow / Keras

MediaPipe

PyAutoGUI

NumPy

How to Run
Install Dependencies:

pip install opencv-python mediapipe tensorflow keras pyautogui numpy
2️⃣ Collect Data (optional, for retraining):

python collect_sequences.py
3️⃣ Train Model:

python train_lstm.py

4️⃣ Run Real-Time Controller:

python gesture_controller.py


Future Improvements

Add more diverse gestures and conditions

Improve edge detection for ultra-wide screens

Build GUI overlay for live feedback

Explore transformer models for richer sequence learning

Integrate with accessibility tools for hands-free control

Conclusion
This project shows how machine learning + computer vision can create intuitive, touchless control systems for everyday tasks.
By combining LSTM-based gesture recognition with real-time GUI automation, we built a system that bridges AI research and practical, user-friendly applications.
