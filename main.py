"""
Main script for Facial Stress Analysis
Simplified menu: train model, run model, exit.
No extra visualization steps.
"""

import os
import json
from datetime import datetime

from data_processing import FacialStressDataProcessor
from train_model import StressModel
from real_time import Detector


class StressAnalysisSystem:
    def __init__(self):
        self.csv_path = 'fer2013.csv'
        self.model_path = 'models/model.h5'
        self.processor = None
        self.cnn = None

    def menu(self):
        while True:
            print('\n' + '=' * 40)
            print('FACIAL STRESS ANALYSIS SYSTEM')
            print('=' * 40)
            print('1. Train model')
            print('2. Run real-time model (webcam)')
            print('3. Exit')

            choice = input('Enter choice (1-3): ').strip()
            if choice == '1':
                self.run_training()
            elif choice == '2':
                self.run_real_time()
            elif choice == '3':
                print('Exiting.')
                break
            else:
                print('Invalid choice. Try again.')

    def run_training(self):
        print('\n--- Training model ---')

        if not os.path.exists(self.csv_path):
            print(f'Error: {self.csv_path} not found. Place FER2013 CSV in the folder.')
            return

        if self.processor is None:
            self.processor = FacialStressDataProcessor(self.csv_path)

        if os.path.exists(self.model_path):
            choice = input('Model exists. [S]kip, [R]etrain, [C]ancel: ').strip().lower()
            if choice == 's':
                print('Skipping training; using existing model.')
                return
            elif choice == 'c':
                print('Training canceled.')
                return
            elif choice == 'r':
                print('Retraining and overwriting model.')
                os.remove(self.model_path)
            else:
                print('Invalid option; training canceled.')
                return

        X_train, y_train, X_val, y_val, _, _ = self.processor.prepare()

        self.cnn = StressModel(input_shape=(48, 48, 1))
        self.cnn.compile()

        try:
            epochs = int(input('Epochs [default 20]: ') or '20')
        except ValueError:
            epochs = 20

        try:
            batch_size = int(input('Batch size [default 64]: ') or '64')
        except ValueError:
            batch_size = 64

        augment_input = input('Use data augmentation? (y/N): ').strip().lower()
        augment = augment_input == 'y'

        print(f'Training model (fixed 20 epochs, batch 64)')
        self.cnn.train(X_train, y_train, X_val, y_val)

        print('Training finished. Saving model...')
        self.cnn.save(self.model_path)

        print('Model saved. You can now run option 2.')

    def run_real_time(self):
        print('\n--- Real-time model ---')

        if not os.path.exists(self.model_path):
            print(f'Error: Model not found at {self.model_path}. Train first.')
            return

        detector = Detector(model_path=self.model_path)
        detector.run(cam=0)


def main():
    system = StressAnalysisSystem()
    system.menu()


if __name__ == '__main__':
    main()
