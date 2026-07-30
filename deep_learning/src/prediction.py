import os
import numpy as np
import tensorflow as tf
from PIL import Image

from deep_learning.src.training import DATA, ARTIFACTS,IMG_SIZE


class Predictor:
    def __init__(self):
        self.classes = sorted(os.listdir(os.path.join(DATA, "train")))

        self.cnn = tf.keras.models.load_model(
            os.path.join(ARTIFACTS, "cnn_model.keras")
        )
        self.tl = tf.keras.models.load_model(
            os.path.join(ARTIFACTS, "tl_model.keras")
        )

    def load_img(self, path):
        img = (
            Image.open(path)
            .convert("RGB")
            .resize(IMG_SIZE)
        ) 

        return np.expand_dims(
            np.array(img, dtype="float32"),
            axis=0,
        ) 

    def _predict(self, model, path):
        prob = model.predict(
            self.load_img(path),
            verbose=0,
        )[0]

        return (
            self.classes[int(np.argmax(prob))],
            round(float(prob.max()) * 100, 2),  
        )

    def predict_cnn(self, path):
        return self._predict(self.cnn, path)

    def predict_tl(self, path):
        return self._predict(self.tl, path)

    def predict_both(self, path):
        return {
            "CNN": self._predict(self.cnn, path),
            "TL": self._predict(self.tl, path),
        }
    
