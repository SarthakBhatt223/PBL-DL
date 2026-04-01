import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class FacialStressDataProcessor:
    """Data processor."""

    STRESS_MAPPING = {0:1,1:1,2:1,3:0,4:0,5:1,6:0}

    def __init__(self, csv_path, image_size=(48,48)):
        self.csv_path = csv_path
        self.image_size = image_size

    def load(self):
        df = pd.read_csv(self.csv_path)
        return df

    def _parse(self, pixels):
        return np.fromstring(pixels, sep=' ', dtype=np.uint8).reshape(self.image_size)

    def _preprocess(self, img):
        return img.astype(np.float32) / 255.0

    def prepare(self):
        df = self.load()

        X = np.array([self._preprocess(self._parse(p)) for p in df['pixels']])
        y = np.array([self.STRESS_MAPPING[e] for e in df['emotion']])

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1, stratify=y_temp, random_state=42
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def save(self, data, path='processed_data'):
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, 'data.npz'), **data)

    @staticmethod
    def load_saved(path):
        return dict(np.load(path))
