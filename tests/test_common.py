import os
import tempfile

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_config
from common.data import (
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    TIME_DIR_MAP,
    TIME_LABELS,
    WEATHER_DIR_MAP,
    WEATHER_LABELS,
    decode_labels,
    image_count,
    is_image_file,
    iter_images,
    onehot,
    preprocess_frame,
)


def test_is_image_file():
    assert is_image_file("a.jpg")
    assert is_image_file("a.JPG")
    assert is_image_file("a.png")
    assert is_image_file("a.jpeg")
    assert not is_image_file("a.txt")
    assert not is_image_file("a.mp4")


def test_image_extensions_tuple():
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS


def test_iter_images_yields_absolute_paths():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "day"))
        for name in ["a.jpg", "b.png", "c.txt", "d.JPEG"]:
            with open(os.path.join(tmp, "day", name), "wb") as f:
                f.write(b"x")
        paths = list(iter_images(os.path.join(tmp, "day")))
        assert len(paths) == 3
        assert all(os.path.isabs(p) for p in paths)
        names = {os.path.basename(p) for p in paths}
        assert names == {"a.jpg", "b.png", "d.JPEG"}


def test_iter_images_limit():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(5):
            with open(os.path.join(tmp, f"{i}.jpg"), "wb") as f:
                f.write(b"x")
        assert len(list(iter_images(tmp))) == 5
        assert len(list(iter_images(tmp, limit=2))) == 2


def test_image_count():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            with open(os.path.join(tmp, f"{i}.jpg"), "wb") as f:
                f.write(b"x")
        with open(os.path.join(tmp, "nope.txt"), "w") as f:
            f.write("x")
        assert image_count(tmp) == 3
    assert image_count("/nonexistent/dir") == 0


def test_decode_labels_from_config():
    cfg = {
        "labels": {"time": ["Dia", "Noche"], "weather": ["Lluvia"]},
        "model": {"input_size": 224},
    }
    decoded = decode_labels(cfg)
    assert decoded["time"] == ["Dia", "Noche"]
    assert decoded["weather"] == ["Lluvia"]
    assert decoded["img_size"] == 224


def test_decode_labels_fallback():
    decoded = decode_labels(None)
    assert decoded["time"] == TIME_LABELS
    assert decoded["weather"] == WEATHER_LABELS
    assert decoded["img_size"] == IMG_SIZE


def test_onehot():
    v = onehot(2, 3)
    assert v.tolist() == [0.0, 0.0, 1.0]


def test_preprocess_frame_shape_and_range():
    import numpy as np

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    out = preprocess_frame(frame, img_size=100)
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.float32
    assert float(out.max()) == pytest.approx(1.0)
    assert float(out.min()) == pytest.approx(1.0)


def test_dir_maps_re_exported():
    assert set(TIME_DIR_MAP.keys()) == set(TIME_LABELS)
    assert set(WEATHER_DIR_MAP.keys()) == set(WEATHER_LABELS)


def test_common_config_is_shared_source():
    assert isinstance(DEFAULT_CONFIG_PATH, str)
    cfg = load_config()
    assert "camera" in cfg