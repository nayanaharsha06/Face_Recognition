# Face Authentication System

## Overview
This project implements a Face Authentication System using OpenCV (Open Source Computer Vision Library) in Python. The system allows users to register their faces, train the system with their facial data, and then authenticate themselves using face recognition.

## Features
- **Face Registration:** Users can register their faces by providing an ID and their name. The system captures multiple images of their face as samples.
- **Face Training:** The captured face samples are used to train the face recognition model using the LBPH (Local Binary Patterns Histograms) algorithm.
- **Face Authentication:** Once trained, the system can authenticate users by recognizing their faces in real-time through a webcam. If the user's face matches the trained data with a certain confidence level, they are authenticated.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)

## Usage
1. Run the `face_authentication.py` script.
2. Follow the on-screen prompts to register your face by providing your ID and name, and capturing face samples.
3. After face registration, the system will prompt you to train the face recognition model.
4. Once trained, the system will continuously authenticate users in real-time through a webcam.

## Note
This project is for educational purposes and demonstrates the implementation of a simple face authentication system using basic face detection and recognition techniques. It may not be suitable for production-level use without further enhancements and security measures.

## Credits
This project is inspired by [https://www.youtube.com/watch?v=LC0p5JmoDI4&t=519s].

