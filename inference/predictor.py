import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2


class Predictor:
    def __init__(self, config):
        self.config = config
        self.model = load_model(config["model"]["path"])
        self.input_size = config["model"]["input_size"]
        self.time_labels = config["labels"]["time"]
        self.weather_labels = config["labels"]["weather"]
        self.smoothing_window = config["inference"]["smoothing_window"]
        self.confidence_threshold = config["inference"]["confidence_threshold"]
        self.model_type = config["model"].get("type", "single_6class")

        self.time_history = []
        self.weather_history = []

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img_to_array(img) / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        x = self.preprocess(frame)

        if self.model_type == "dual_head":
            preds = self.model.predict(x, verbose=0)
            time_probs = preds[0][0]
            weather_probs = preds[1][0]
        else:
            preds = self.model.predict(x, verbose=0)[0]
            time_probs = preds[:3]
            weather_probs = preds[3:]

        time_idx = int(np.argmax(time_probs))
        weather_idx = int(np.argmax(weather_probs))
        time_conf = float(np.max(time_probs))
        weather_conf = float(np.max(weather_probs))

        self.time_history.append(time_idx)
        self.weather_history.append(weather_idx)

        if len(self.time_history) > self.smoothing_window:
            self.time_history.pop(0)
        if len(self.weather_history) > self.smoothing_window:
            self.weather_history.pop(0)

        smoothed_time = max(set(self.time_history), key=self.time_history.count)
        smoothed_weather = max(set(self.weather_history), key=self.weather_history.count)

        result = {
            "time_label": self.time_labels[smoothed_time],
            "time_confidence": time_conf * 100,
            "weather_label": self.weather_labels[smoothed_weather],
            "weather_confidence": weather_conf * 100,
            "time_probs": time_probs.tolist(),
            "weather_probs": weather_probs.tolist(),
        }
        return result
