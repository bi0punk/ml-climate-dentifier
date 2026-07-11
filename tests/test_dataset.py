import os
import tempfile
import numpy as np
from PIL import Image

from training.dataset import (
    load_dataset,
    DualDataGenerator,
    onehot,
    TIME_LABELS,
    WEATHER_LABELS,
)


def create_dummy_dataset(tmpdir):
    dirs = {
        "day": "day",
        "evening": "evening",
        "night (Nightvision)": "night",
        "clear": "clear",
        "cloudy": "cloudy",
        "partly_cloudy": "partly_cloudy",
    }
    for src_name, dst_name in dirs.items():
        path = os.path.join(tmpdir, dst_name)
        os.makedirs(path, exist_ok=True)
        img = Image.fromarray(np.ones((150, 150, 3), dtype=np.uint8) * 128)
        img.save(os.path.join(path, f"{dst_name}_001.jpg"))
        if dst_name != src_name:
            img.save(os.path.join(path, f"{dst_name}_002.jpg"))


def test_onehot():
    v = onehot(0, 3)
    assert v[0] == 1.0 and v[1] == 0.0 and v[2] == 0.0
    v = onehot(2, 5)
    assert v[2] == 1.0 and sum(v) == 1.0


def test_load_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        images, time_labels, weather_labels, time_known, weather_known = load_dataset(tmpdir)

        assert len(images) > 0
        assert len(time_labels) == len(images)
        assert len(weather_labels) == len(images)

        time_dir_count = 0
        weather_dir_count = 0
        for i in range(len(images)):
            if time_known[i] == 1.0:
                time_dir_count += 1
                assert np.sum(time_labels[i]) == 1.0
            if weather_known[i] == 1.0:
                weather_dir_count += 1
                assert np.sum(weather_labels[i]) == 1.0

        assert time_dir_count > 0
        assert weather_dir_count > 0


def test_dual_data_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        images, time_labels, weather_labels, time_known, weather_known = load_dataset(tmpdir)

        gen = DualDataGenerator(
            images, time_labels, weather_labels, time_known, weather_known,
            batch_size=4, augment=False,
        )

        assert len(gen) >= 1
        X, y, w = gen[0]
        assert X.shape[0] <= 4
        assert X.shape[1:] == (150, 150, 3)
        assert "time" in y
        assert "weather" in y
        assert "time" in w
        assert "weather" in w
        assert y["time"].shape[1] == len(TIME_LABELS)
        assert y["weather"].shape[1] == len(WEATHER_LABELS)
