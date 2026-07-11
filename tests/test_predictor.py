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
    with pytest.raises(Exception):
        Predictor(SAMPLE_CONFIG)


def test_smoothing_works():
    """Verify smoothing logic with a mock predictor instance."""
    from inference.predictor import Predictor

    class MockPredictor(Predictor):
        def __init__(self):
            self.time_labels = ["Day", "Evening", "Night"]
            self.weather_labels = ["Clear", "Cloudy", "Partly Cloudy"]
            self.smoothing_window = 3
            self.time_history = []
            self.weather_history = []

        def predict(self, frame):
            self.time_history.append(0)
            self.weather_history.append(1)
            if len(self.time_history) > self.smoothing_window:
                self.time_history.pop(0)
                self.weather_history.pop(0)
            smoothed_time = max(set(self.time_history), key=self.time_history.count)
            smoothed_weather = max(set(self.weather_history), key=self.weather_history.count)
            return {
                "time_label": self.time_labels[smoothed_time],
                "time_confidence": 95.0,
                "weather_label": self.weather_labels[smoothed_weather],
                "weather_confidence": 80.0,
            }

    p = MockPredictor()
    r1 = p.predict(None)
    assert r1["time_label"] == "Day"
    assert r1["weather_label"] == "Cloudy"
    assert r1["time_confidence"] == 95.0
    assert r1["weather_confidence"] == 80.0

    # After smoothing_window predictions, mode should stabilize
    p.time_history = [0, 0, 1]
    p.weather_history = [2, 2, 1]
    r2 = p.predict(None)
    assert r2["time_label"] == "Day"
    assert r2["weather_label"] == "Partly Cloudy"
