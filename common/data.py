"""Decodificación de clases (labels) y helpers de imágenes compartidos.

Unifica la fuente de verdad para labels de tiempo/clima y el preprocesado de
imágenes, de modo que training, inference y scripts no dupliquen lógica.

- Las etiquetas canónicas vienen de ``config.yaml`` (decodificadas de forma
  consistente) y de ``constants`` para el mapeo a directorios del dataset.
"""

import os

import numpy as np

from constants import IMG_SIZE, TIME_DIR_MAP, TIME_LABELS, WEATHER_DIR_MAP, WEATHER_LABELS

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_image_file(fname):
    return fname.lower().endswith(IMAGE_EXTENSIONS)


def list_images(path):
    """Devuelve rutas de imágenes válidas dentro de un directorio."""
    return [
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if is_image_file(fname)
    ]


def image_count(path):
    """Cuenta imágenes dentro de un directorio (0 si no existe)."""
    if not os.path.isdir(path):
        return 0
    return len(list_images(path))


def decode_labels(config=None):
    """Decodifica labels de tiempo/clima desde config.yaml con fallback a constants.

    Devuelve un dict con ``time``, ``weather`` (etiquetas) y ``img_size``.
    Usado por predictor y por módulos de entrenamiento que leen config.
    """
    if config is not None:
        labels = config.get("labels", {})
        time_labels = labels.get("time", TIME_LABELS)
        weather_labels = labels.get("weather", WEATHER_LABELS)
        img_size = config.get("model", {}).get("input_size", IMG_SIZE)
    else:
        time_labels, weather_labels, img_size = TIME_LABELS, WEATHER_LABELS, IMG_SIZE

    return {
        "time": time_labels,
        "weather": weather_labels,
        "img_size": img_size,
    }


def get_time_dir_map(config=None):
    """Mapeo time-label -> nombre de directorio (base en constants)."""
    return TIME_DIR_MAP


def get_weather_dir_map(config=None):
    """Mapeo weather-label -> nombre de directorio (base en constants)."""
    return WEATHER_DIR_MAP


def preprocess_image(path, img_size=IMG_SIZE):
    """Carga una imagen desde disco, la reescala y normaliza a [0, 1].

    Usado por el pipeline de entrenamiento (Sequence).
    """
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    img = load_img(path, target_size=(img_size, img_size))
    return img_to_array(img) / 255.0


def preprocess_frame(frame, img_size=IMG_SIZE):
    """Preprocesa un frame (numpy array BGR) a entrada del modelo.

    Usado por el predictor en tiempo real.
    """
    import cv2
    from tensorflow.keras.preprocessing.image import img_to_array

    img = cv2.resize(frame, (img_size, img_size))
    return img_to_array(img) / 255.0


def to_batch(x):
    """Añade dimensión batch a un tensor preprocesado."""
    return np.expand_dims(x, axis=0)


def onehot(idx, size):
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v