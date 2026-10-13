import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from common.data import (
    IMG_SIZE,
    TIME_DIR_MAP,
    TIME_LABELS,
    WEATHER_DIR_MAP,
    WEATHER_LABELS,
    is_image_file,
    onehot,
    preprocess_image,
)

RAW_DIR = "dataset"


def load_dataset(raw_dir=RAW_DIR):
    """Recorre el dataset raw (directorios por clase) y genera tensores de
    labels OneHot + máscaras de "known" para el dual-head.

    Cada imagen aporta solo un tipo de etiqueta conocida (tiempo O clima),
    el otro se deja en ceros junto con su máscara ``known=0``.
    """
    images = []
    time_labels = []
    weather_labels = []
    time_known = []
    weather_known = []

    for time_label, time_dir in TIME_DIR_MAP.items():
        path = os.path.join(raw_dir, time_dir)
        if not os.path.isdir(path):
            continue
        time_idx = TIME_LABELS.index(time_label)
        for fname in os.listdir(path):
            if not is_image_file(fname):
                continue
            images.append(os.path.join(path, fname))
            time_labels.append(onehot(time_idx, len(TIME_LABELS)))
            weather_labels.append(np.zeros(len(WEATHER_LABELS), dtype=np.float32))
            time_known.append(1.0)
            weather_known.append(0.0)

    for weather_label, weather_dir in WEATHER_DIR_MAP.items():
        path = os.path.join(raw_dir, weather_dir)
        if not os.path.isdir(path):
            continue
        weather_idx = WEATHER_LABELS.index(weather_label)
        for fname in os.listdir(path):
            if not is_image_file(fname):
                continue
            images.append(os.path.join(path, fname))
            time_labels.append(np.zeros(len(TIME_LABELS), dtype=np.float32))
            weather_labels.append(onehot(weather_idx, len(WEATHER_LABELS)))
            time_known.append(0.0)
            weather_known.append(1.0)

    return (
        np.array(images),
        np.array(time_labels),
        np.array(weather_labels),
        np.array(time_known),
        np.array(weather_known),
    )


class DualDataGenerator(Sequence):
    def __init__(self, image_paths, time_labels, weather_labels, time_known, weather_known, batch_size=32, augment=False):
        self.image_paths = image_paths
        self.time_labels = time_labels
        self.weather_labels = weather_labels
        self.time_known = time_known
        self.weather_known = weather_known
        self.batch_size = batch_size
        self.augment = augment
        self.n = len(image_paths)
        self.indices = np.arange(self.n)
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        ) if augment else None

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n)
        batch_indices = self.indices[start:end]
        batch_size = len(batch_indices)

        X = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        y_time = np.zeros((batch_size, len(TIME_LABELS)), dtype=np.float32)
        y_weather = np.zeros((batch_size, len(WEATHER_LABELS)), dtype=np.float32)
        w_time = np.zeros(batch_size, dtype=np.float32)
        w_weather = np.zeros(batch_size, dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            X[i] = preprocess_image(self.image_paths[idx])
            y_time[i] = self.time_labels[idx]
            y_weather[i] = self.weather_labels[idx]
            w_time[i] = self.time_known[idx]
            w_weather[i] = self.weather_known[idx]

        if self.datagen is not None:
            X = self.datagen.flow(X, shuffle=False, batch_size=batch_size)[0]

        return (X,
                {"time": y_time, "weather": y_weather},
                {"time": w_time, "weather": w_weather})

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def _make_generator(idx_list, images, time_labels, weather_labels, time_known, weather_known, augment, batch_size):
    """Construye un DualDataGenerator sobre un subconjunto de índices."""
    return DualDataGenerator(
        image_paths=images[idx_list],
        time_labels=time_labels[idx_list],
        weather_labels=weather_labels[idx_list],
        time_known=time_known[idx_list],
        weather_known=weather_known[idx_list],
        augment=augment,
        batch_size=batch_size,
    )


def create_splits(raw_dir=RAW_DIR, test_size=0.15, val_size=0.15, random_state=42, batch_size=32):
    images, time_labels, weather_labels, time_known, weather_known = load_dataset(raw_dir)

    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1 - test_size), random_state=random_state)

    return (
        _make_generator(train_idx, images, time_labels, weather_labels, time_known, weather_known,
                        augment=True, batch_size=batch_size),
        _make_generator(val_idx, images, time_labels, weather_labels, time_known, weather_known,
                        augment=False, batch_size=batch_size),
        _make_generator(test_idx, images, time_labels, weather_labels, time_known, weather_known,
                        augment=False, batch_size=batch_size),
    )


def get_train_val_generators(raw_dir=RAW_DIR, batch_size=32):
    train_gen, val_gen, _ = create_splits(raw_dir, batch_size=batch_size)
    return train_gen, val_gen