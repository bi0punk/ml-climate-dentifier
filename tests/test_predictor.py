import numpy as np
import pytest

from inference.predictor import Predictor

SAMPLE_CONFIG = {
    "model": {
        "path": "nonexistent.h5",
        "input_size": 150,
        "type": "single_6class",
    },
    "inference": {
        "smoothing_window": 3,
        "confidence_threshold": 0.0,
    },
    "labels": {
        "time": ["Day", "Evening", "Night"],
        "weather": ["Clear", "Cloudy", "Partly Cloudy"],
    },
}


def test_predictor_init_fails_without_model():
    with pytest.raises(FileNotFoundError):
        Predictor(SAMPLE_CONFIG)


def _make_predictor():
    p = Predictor.__new__(Predictor)
    p.input_size = 150
    p.time_labels = ["Day", "Evening", "Night"]
    p.weather_labels = ["Clear", "Cloudy", "Partly Cloudy"]
    p.smoothing_window = 3
    p.confidence_threshold = 0.0
    p.time_history = []
    p.weather_history = []
    return p


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_smoothing_works():
    """Verifica el suavizado temporal en la ventana de moda."""
    p = _make_predictor()

    class FakeModel:
        def predict(self, x, verbose=0):
            return (np.array([[0.9, 0.05, 0.05]]), np.array([[0.1, 0.8, 0.1]]))

    p.model = FakeModel()
    p.model_type = "dual_head"

    r1 = p.predict(_frame())
    assert r1["time_label"] == "Day"
    assert r1["weather_label"] == "Cloudy"
    assert r1["time_confidence"] == pytest.approx(90.0)
    assert r1["weather_confidence"] == pytest.approx(80.0)

    # La ventana de suavizado no debe superar smoothing_window
    assert len(p.time_history) == 1
    assert len(p.weather_history) == 1


def test_smoothing_window_bounded():
    """Tras más predicciones que la ventana, el historial queda acotado."""
    p = _make_predictor()

    class FakeModel:
        def predict(self, x, verbose=0):
            return (np.array([[0.9, 0.05, 0.05]]), np.array([[0.1, 0.1, 0.8]]))

    p.model = FakeModel()
    p.model_type = "dual_head"

    for _ in range(10):
        p.predict(_frame())

    assert len(p.time_history) == p.smoothing_window
    assert len(p.weather_history) == p.smoothing_window
    # la moda con mayoría "Partly Cloudy" debe estabilizarse
    assert p.predict(_frame())["weather_label"] == "Partly Cloudy"


def test_single_6class_output_slices():
    """Modelo single_6class: primeros 3 = time, últimos 3 = weather."""
    p = _make_predictor()

    class FakeModel:
        def predict(self, x, verbose=0):
            return np.array([[0.1, 0.8, 0.1, 0.2, 0.2, 0.6]])

    p.model = FakeModel()
    p.model_type = "single_6class"

    r = p.predict(_frame())
    assert r["time_label"] == "Evening"
    assert r["weather_label"] == "Partly Cloudy"


def test_preprocess_normalizes_to_01():
    """preprocess debe producir un batch [0,1] normalizado."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    p = _make_predictor()
    p.input_size = 150
    out = p.preprocess(frame)
    assert out.shape == (1, 150, 150, 3)
    assert out.dtype == np.float32
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0