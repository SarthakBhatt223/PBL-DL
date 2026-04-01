# Facial Stress Analysis - Mental Readiness Assessment (Simplified)

This repository implements a simplified facial stress detection pipeline using CNN + webcam inference. It matches current code state.

## Modules

1. data_processing.py - loads fer2013.csv, parses pixel strings, maps to stress labels, splits via prepare().
2. train_model.py - defines StressModel (compile/train/evaluate/save), saves model to models/model.h5.
3. real_time.py - defines Detector (run(cam=0)), loads saved model, performs webcam inference.
4. main.py - simple menu: 1) Train model, 2) Run real-time model (webcam), 3) Exit.

## Quick start

1. Create venv and install: pip install tensorflow opencv-python numpy pandas scikit-learn
2. Place fer2013.csv in root.
3. Run: python main.py

## Workflow

- Option 1 trains and saves model to models/model.h5.
- Option 2 uses saved model for real-time webcam inference.
- Option 3 exits.
