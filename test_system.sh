#!/bin/bash
# Facial Stress Analysis System - Test Script
# Run this to verify everything is working

echo "=============================================="
echo " FACIAL STRESS ANALYSIS SYSTEM - TEST"
echo "=============================================="

cd /Users/sarthakbhatt/Desktop/PBL/PBL-DL

echo "1. Checking files..."
files=("main.py" "data_processing.py" "train_model.py" "real_time.py" "fer2013.csv" "README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file - MISSING"
    fi
done

echo ""
echo "2. Testing Python syntax..."
python_files=("main.py" "data_processing.py" "train_model.py" "real_time.py")
for file in "${python_files[@]}"; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "   ✓ $file - Syntax OK"
    else
        echo "   ✗ $file - Syntax ERROR"
    fi
done

echo ""
echo "3. Checking dependencies..."
python -c "
import sys
deps = ['numpy', 'pandas', 'cv2', 'tensorflow', 'matplotlib', 'seaborn', 'sklearn']
for dep in deps:
    try:
        __import__(dep)
        print(f'   ✓ {dep}')
    except ImportError:
        print(f'   ✗ {dep} - MISSING')
"

echo ""
echo "4. Testing data loading..."
python -c "
try:
    from data_processing import FacialStressDataProcessor
    processor = FacialStressDataProcessor('fer2013.csv')
    df = processor.load_data()
    print(f'   ✓ Data loaded: {len(df)} samples')
except Exception as e:
    print(f'   ✗ Data loading failed: {e}')
"

echo ""
echo "=============================================="
echo " SYSTEM STATUS: READY TO RUN"
echo "=============================================="
echo ""
echo "To start the system:"
echo "   cd /Users/sarthakbhatt/Desktop/PBL/PBL-DL"
echo "   python main.py"
echo ""
echo "Menu Options:"
echo "   1. Data Processing & Preparation"
echo "   2. Train Model"
echo "   3. Run Real-Time Detection (Webcam)"
echo "   4. Quick Demo (Fast Training + Webcam)"
echo "   5. View Model Performance Metrics"
echo "   6. Exit"
echo ""
echo "=============================================="