import os
import tempfile

import numpy as np
from PIL import Image

from common.data import onehot
from training.dataset import (
    TIME_LABELS,
    WEATHER_LABELS,
    DualDataGenerator,
    create_splits,
    get_train_val_generators,
    load_dataset,
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


def _load(tmpdir):
    return load_dataset(tmpdir)


def test_onehot():
    v = onehot(0, 3)
    assert v[0] == 1.0 and v[1] == 0.0 and v[2] == 0.0
    v = onehot(2, 5)
    assert v[2] == 1.0 and sum(v) == 1.0


def test_load_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        images, time_labels, weather_labels, time_known, weather_known = _load(tmpdir)

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


def test_load_dataset_accepts_case_insensitive_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "day"))
        img = Image.fromarray(np.zeros((150, 150, 3), dtype=np.uint8))
        img.save(os.path.join(tmpdir, "day", "ok.jpg"))
        img.save(os.path.join(tmpdir, "day", "upper.JPG"))
        with open(os.path.join(tmpdir, "day", "skip.txt"), "w") as f:
            f.write("x")

        images, _, _, _, _ = _load(tmpdir)
        # .jpg/.JPG admitidos, .txt ignorado
        assert len(images) == 2


def test_load_dataset_missing_dirs_ok():
    with tempfile.TemporaryDirectory() as tmpdir:
        images, time_labels, weather_labels, time_known, weather_known = _load(tmpdir)
        assert len(images) == 0
        assert len(time_labels) == 0
        assert len(weather_labels) == 0


def test_dual_data_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        images, time_labels, weather_labels, time_known, weather_known = _load(tmpdir)

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


def test_dual_data_generator_shuffles_on_epoch_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        images, time_labels, weather_labels, time_known, weather_known = _load(tmpdir)

        gen = DualDataGenerator(
            images, time_labels, weather_labels, time_known, weather_known,
            batch_size=len(images), augment=False,
        )
        before = gen.indices.copy()
        gen.on_epoch_end()
        if len(images) > 1:
            assert not np.array_equal(before, gen.indices)
        # El orden sigue siendo una permutación
        assert sorted(gen.indices.tolist()) == list(range(len(images)))


def test_dual_data_generator_augment_keeps_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        images, time_labels, weather_labels, time_known, weather_known = _load(tmpdir)

        gen = DualDataGenerator(
            images, time_labels, weather_labels, time_known, weather_known,
            batch_size=4, augment=True,
        )
        X, y, w = gen[0]
        assert X.shape[0] <= 4
        assert X.shape[1:] == (150, 150, 3)
        # La augmentación aplica roto/zoom/brightness pero mantiene el rango
        assert X.dtype == np.float32


def test_create_splits():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        train_gen, val_gen, test_gen = create_splits(tmpdir, test_size=0.15, val_size=0.15, random_state=42, batch_size=4)

        assert len(train_gen) >= 1
        assert len(val_gen) >= 1
        assert len(test_gen) >= 1

        X, y, w = val_gen[0]
        assert X.shape[1:] == (150, 150, 3)
        assert set(y.keys()) == {"time", "weather"}
        assert set(w.keys()) == {"time", "weather"}


def test_create_splits_reproducible():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        splits1 = create_splits(tmpdir, random_state=42)
        splits2 = create_splits(tmpdir, random_state=42)
        assert len(splits1) == len(splits2) == 3
        for gen1, gen2 in zip(splits1, splits2):
            assert np.array_equal(gen1.indices, gen2.indices)


def test_load_dataset_limit_per_class():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "day")
        os.makedirs(path)
        img = Image.fromarray(np.zeros((150, 150, 3), dtype=np.uint8))
        for i in range(4):
            img.save(os.path.join(path, f"img{i}.jpg"))

        full = len(_load(tmpdir)[0])
        limited = len(load_dataset(tmpdir, limit=2)[0])
        assert full == 4
        assert limited == 2


def test_get_train_val_generators():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        train_gen, val_gen = get_train_val_generators(tmpdir, batch_size=4)
        assert isinstance(train_gen, DualDataGenerator)
        assert isinstance(val_gen, DualDataGenerator)
        assert train_gen.augment is True
        assert val_gen.augment is False