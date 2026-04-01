"""
Real-time Facial Stress Detection using Webcam
Detects and displays stress/mental readiness status in real-time
"""

import cv2
import numpy as np
from tensorflow import keras


class Detector:
    def __init__(self, model_path='models/model.h5'):
        self.model = keras.models.load_model(model_path)
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48,48)).astype(np.float32) / 255.0
        return img[np.newaxis, ..., np.newaxis]

    def _predict(self, face):
        p = self.model.predict(self._preprocess(face), verbose=0)[0][0]
        return ("STRESSED", p) if p > 0.5 else ("RELAXED", 1-p)

    def run(self, cam=0):
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face.detectMultiScale(gray,1.1,5)

            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                label, conf = self._predict(roi)

                color = (0,0,255) if label=="STRESSED" else (0,255,0)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,f"{label} {conf:.2f}",
                            (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,color,2)

            cv2.imshow("Stress",frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    Detector().run()


if __name__ == "__main__":
    main()
