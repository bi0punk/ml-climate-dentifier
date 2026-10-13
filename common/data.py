"""Decodificación de clases (labels) y helpers de imágenes compartidos.

Unifica la fuente de verdad para labels de tiempo/clima y el preprocesado de
imágenes, de modo que training, inference y scripts no dupliquen lógica.

- Las etiquetas canónicas vienen de ``config.yaml`` (decodificadas de forma
  consistente) y de ``constants`` para el mapeo a directorios del dataset.
"""

import os

import numpy as np

from constants import (
    IMG_SIZE as IMG_SIZE,
)
from constants import (
    TIME_DIR_MAP as TIME_DIR_MAP,
)
from constants import (
    TIME_LABELS as TIME_LABELS,
)
from constants import (
    WEATHER_DIR_MAP as WEATHER_DIR_MAP,
)
from constants import (
    WEATHER_LABELS as WEATHER_LABELS,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_image_file(fname):
    return fname.lower().endswith(IMAGE_EXTENSIONS)


def iter_images(path, limit=None):
    """Itera rutas de imágenes válidas en un directorio.

    Lee el árbol de forma perezosa: si se indica ``limit``, detiene el
    recorrido al alcanzar ese número (evita cargar directorios enormes).
    """
    count = 0
    for fname in os.listdir(path):
        if not is_image_file(fname):
            continue
        yield os.path.join(path, fname)
        count += 1
        if limit is not None and count >= limit:
            return


def image_count(path):
    """Cuenta imágenes dentro de un directorio (0 si no existe)."""
    if not os.path.isdir(path):
        return 0
    return sum(1 for _ in iter_images(path))


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


def onehot(idx, size):
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v