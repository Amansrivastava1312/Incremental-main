import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "dataset")
ARTIFACTS = os.path.join(BASE, "artifacts")
IMG_SIZE = (160, 160)
BATCH_SIZE = 32


class Trainer:
    def load(self, split):
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATA, split), shuffle=True,
            batch_size=BATCH_SIZE, image_size=IMG_SIZE)

    def build_cnn(self, n):
        model = models.Sequential([
            layers.Rescaling(1. / 255, input_shape=(160, 160, 3)),
            layers.Conv2D(32, 3, activation='relu'), 
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'), 
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'), 
            layers.Dropout(0.5),
            layers.Dense(n, activation='softmax')])
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_tl(self, n):
        base = applications.ResNet50V2(weights='imagenet', include_top=False,
                                       input_shape=(160, 160, 3))
        base.trainable = False
        model = models.Sequential([
            layers.Rescaling(1. / 255),
            base,
            layers.Flatten(),
            layers.Dense(16, activation='relu'), layers.Dropout(0.6),
            layers.Dense(n, activation='softmax')])
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, model, train_ds, val_ds, epochs=3):
        return model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def save(self, model, name):
        os.makedirs(ARTIFACTS, exist_ok=True)
        model.save(os.path.join(ARTIFACTS, name))