import json
import os
import sys
import tempfile

import build_dataset
from label_captures import infer_time_of_day, process_captures


def _make_dirs(root, names):
    for name in names:
        path = os.path.join(root, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "sample.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xe0junk")
    return root


# ---- build_dataset ----

def test_analyze_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        _make_dirs(tmp, ["day", "clear"])
        stats = build_dataset.analyze_dataset(tmp)
        assert stats == {"day": 1, "clear": 1}


def test_analyze_dataset_empty_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "day"))
        stats = build_dataset.analyze_dataset(tmp)
        assert stats == {"day": 0}


def test_analyze_dataset_ignores_non_images():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "day")
        os.makedirs(path)
        with open(os.path.join(path, "a.jpg"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(path, "b.txt"), "w") as f:
            f.write("x")
        stats = build_dataset.analyze_dataset(tmp)
        assert stats["day"] == 1


def test_build_combined_dataset_limit():
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out = os.path.join(tmp, "out")
        day_path = os.path.join(raw, "day")
        os.makedirs(day_path, exist_ok=True)
        for i in range(5):
            with open(os.path.join(day_path, f"img{i}.jpg"), "wb") as f:
                f.write(b"x")

        counts = build_dataset.build_combined_dataset(raw, out, symlink=False, limit=2)
        # solo 2 de 5 imágenes de day se propagan a cada clase de clima
        assert counts["day_clear"] == 2
        assert all(v == 2 for v in counts.values())


def test_build_combined_dataset_symlink():
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out = os.path.join(tmp, "out")
        _make_dirs(raw, ["day", "evening", "night (Nightvision)", "clear", "cloudy", "partly_cloudy"])

        counts = build_dataset.build_combined_dataset(raw, out, symlink=True)
        expected = {f"{t}_{w}": 2 for t in ["day", "evening", "night"] for w in ["clear", "cloudy", "partly_cloudy"]}
        assert counts == expected
        # Los enlaces simbólicos apuntan a las fuentes
        cls_dir = os.path.join(out, "day_clear")
        assert os.path.islink(os.path.join(cls_dir, "sample.jpg"))


def test_build_combined_dataset_copy():
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out = os.path.join(tmp, "out")
        _make_dirs(raw, ["day", "clear"])
        os.makedirs(os.path.join(raw, "day_extra"), exist_ok=True)
        # añadimos un segundo fname para forzar dedupe dentro de day_*
        with open(os.path.join(raw, "day_extra", "other.jpg"), "wb") as f:
            f.write(b"x")

        counts = build_dataset.build_combined_dataset(raw, out, symlink=False)
        assert counts["day_clear"] >= 2
        cls_dir = os.path.join(out, "day_clear")
        assert os.path.exists(os.path.join(cls_dir, "sample.jpg"))
        assert not os.path.islink(os.path.join(cls_dir, "sample.jpg"))


def test_build_negative_class_counts():
    """direcciones de clima producen clases para cada time_label."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out = os.path.join(tmp, "out")
        _make_dirs(raw, ["clear"])  # solo clima, sin time dirs
        counts = build_dataset.build_combined_dataset(raw, out, symlink=False)
        # 1 imagen clear -> 3 clases time (misma imagen)
        assert counts == {
            "day_clear": 1, "evening_clear": 1, "night_clear": 1,
        }


def test_main_analyze_prints_stats(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        _make_dirs(tmp, ["day"])
        old = sys.argv
        try:
            sys.argv = ["build_dataset.py", "--raw-dir", tmp, "--analyze"]
            build_dataset.main()
        finally:
            sys.argv = old
        out = capsys.readouterr().out
        assert "Dataset Analysis" in out
        assert "day" in out


def test_main_build_writes_info_json():
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out = os.path.join(tmp, "out")
        _make_dirs(raw, ["day", "clear"])
        old = sys.argv
        try:
            sys.argv = ["build_dataset.py", "--raw-dir", raw, "--output-dir", out, "--copy"]
            build_dataset.main()
        finally:
            sys.argv = old

        info_path = os.path.join(out, "dataset_info.json")
        assert os.path.exists(info_path)
        with open(info_path) as f:
            info = json.load(f)
        assert "classes" in info
        assert "class_counts" in info
        assert "day_clear" in info["classes"]


# ---- label_captures ----

def test_infer_time_of_day():
    assert infer_time_of_day(5) == "night"
    assert infer_time_of_day(6) == "day"
    assert infer_time_of_day(11) == "day"
    assert infer_time_of_day(12) == "day"
    assert infer_time_of_day(17) == "day"
    assert infer_time_of_day(18) == "evening"
    assert infer_time_of_day(20) == "evening"
    assert infer_time_of_day(21) == "night"
    assert infer_time_of_day(23) == "night"


def test_process_captures_symlink():
    with tempfile.TemporaryDirectory() as tmp:
        cap = os.path.join(tmp, "captures")
        out = os.path.join(tmp, "out")
        os.makedirs(cap)
        files = {
            "capture_20260714_100000.jpg": "day",
            "capture_20260714_191500.jpg": "evening",
            "capture_20260715_020000.png": "night",
            "capture_20260715_030000.jpeg": "night",
        }
        for fname in files:
            with open(os.path.join(cap, fname), "wb") as f:
                f.write(b"\xff\xd8")

        stats = process_captures(cap, out)
        assert stats["day"] == 1
        assert stats["evening"] == 1
        assert stats["night"] == 2
        assert stats["unknown"] == 0
        assert stats["skipped"] == 0

        for label in {"day", "evening", "night"}:
            for fname in files:
                if files[fname] == label:
                    assert os.path.islink(os.path.join(out, label, fname))


def test_process_captures_copy_and_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        cap = os.path.join(tmp, "captures")
        out = os.path.join(tmp, "out")
        os.makedirs(cap)
        with open(os.path.join(cap, "capture_badname.jpg"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(cap, "notes.txt"), "w") as f:
            f.write("x")
        with open(os.path.join(cap, "capture_20260714_100000.jpg"), "wb") as f:
            f.write(b"x")

        stats = process_captures(cap, out, copy=True)
        assert stats["unknown"] == 1
        assert stats["skipped"] == 1
        assert stats["day"] == 1
        dst = os.path.join(out, "day", "capture_20260714_100000.jpg")
        assert os.path.exists(dst)
        assert not os.path.islink(dst)