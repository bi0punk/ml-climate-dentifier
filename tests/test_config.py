import os
import tempfile

import yaml

from common.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_exists():
    assert os.path.exists(DEFAULT_CONFIG_PATH), f"Config not found: {DEFAULT_CONFIG_PATH}"


def test_load_default_config():
    cfg = load_config()
    assert "camera" in cfg
    assert "model" in cfg
    assert "inference" in cfg
    assert "display" in cfg
    assert "record" in cfg
    assert "labels" in cfg

    assert cfg["camera"]["url"] == "${RTSP_URL}"
    assert cfg["model"]["input_size"] == 150
    assert cfg["inference"]["frame_skip"] == 5
    assert cfg["labels"]["time"] == ["Day", "Evening", "Night"]
    assert cfg["labels"]["weather"] == ["Clear", "Cloudy", "Partly Cloudy"]


def test_env_override(monkeypatch):
    monkeypatch.setenv("IP_CAMERA_URL", "rtsp://test:8554/stream")
    monkeypatch.setenv("MODEL_PATH", "models/test.h5")
    cfg = load_config()
    assert cfg["camera"]["url"] == "rtsp://test:8554/stream"
    assert cfg["model"]["path"] == "models/test.h5"


def test_env_override_precedence_over_yaml(monkeypatch):
    data = {
        "camera": {"url": "http://yaml:81/stream"},
        "model": {"path": "models/yaml.h5", "input_size": 224},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        monkeypatch.setenv("IP_CAMERA_URL", "rtsp://env:554/x")
        monkeypatch.setenv("MODEL_PATH", "models/env.h5")
        cfg = load_config(path)
        assert cfg["camera"]["url"] == "rtsp://env:554/x"
        assert cfg["model"]["path"] == "models/env.h5"
        assert cfg["model"]["input_size"] == 224
    finally:
        os.unlink(path)


def test_load_custom_config():
    data = {
        "camera": {"url": "http://custom:81/stream"},
        "model": {"path": "models/custom.h5", "input_size": 224},
        "inference": {"frame_skip": 10, "smoothing_window": 3, "confidence_threshold": 0.5},
        "display": {"width": 1280, "height": 720},
        "record": {"enabled": True, "output_path": "test.avi"},
        "labels": {"time": ["Day", "Night"], "weather": ["Clear", "Rain"]},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        cfg = load_config(f.name)
    os.unlink(f.name)

    assert cfg["camera"]["url"] == "http://custom:81/stream"
    assert cfg["model"]["input_size"] == 224
    assert cfg["inference"]["frame_skip"] == 10


def test_load_config_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_delegation_from_inference_config():
    """inference.config debe re-exportar el cargador compartido."""
    from inference import config as inference_config

    assert inference_config.load_config is load_config
    assert inference_config.DEFAULT_CONFIG_PATH == DEFAULT_CONFIG_PATH