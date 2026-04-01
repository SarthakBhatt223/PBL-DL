"""
Model training script for Facial Stress Analysis
Builds and trains a CNN to classify stress/not-stressed emotions
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score
from data_processing import FacialStressDataProcessor


class StressModel:
    def __init__(self, input_shape=(48,48,1)):
        self.model = self._build(input_shape)

    def _build(self, shape):
        return keras.Sequential([
            layers.Conv2D(32,3,activation='relu',input_shape=shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64,3,activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128,activation='relu'),
            layers.Dense(1,activation='sigmoid')
        ])

    def compile(self):
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val):
        if len(X_train.shape)==3:
            X_train=X_train[...,np.newaxis]
            X_val=X_val[...,np.newaxis]

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val,y_val),
            epochs=20,
            batch_size=64,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        if len(X_test.shape)==3:
            X_test=X_test[...,np.newaxis]

        loss, acc = self.model.evaluate(X_test,y_test,verbose=0)
        preds = (self.model.predict(X_test)>0.5).astype(int).flatten()
        f1 = f1_score(y_test,preds)

        return {"loss":loss,"accuracy":acc,"f1":f1}

    def save(self, path='models/model.h5'):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        self.model.save(path)


def main():
    if not os.path.exists('fer2013.csv'):
        return

    dp = FacialStressDataProcessor('fer2013.csv')
    X_train,y_train,X_val,y_val,X_test,y_test = dp.prepare()

    model = StressModel()
    model.compile()
    model.train(X_train,y_train,X_val,y_val)

    metrics = model.evaluate(X_test,y_test)
    model.save()

    print(metrics)


if __name__ == "__main__":
    main()
