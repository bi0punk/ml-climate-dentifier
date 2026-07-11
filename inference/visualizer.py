import cv2
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, config):
        self.config = config["display"]
        self.record_config = config["record"]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.out = None
        self._setup_recorder()

    def _setup_recorder(self):
        if self.record_config["enabled"]:
            fourcc = cv2.VideoWriter_fourcc(*self.record_config["codec"])
            w, h = self.config["width"], self.config["height"]
            self.out = cv2.VideoWriter(
                self.record_config["output_path"],
                fourcc,
                self.record_config["fps"],
                (w, h),
            )
            logger.info(f"Recording to {self.record_config['output_path']}")

    def draw(self, frame, prediction):
        font_scale = self.config["font_scale"]
        thickness = self.config["font_thickness"]
        line_type = cv2.LINE_AA

        texts = [
            (f"Time: {prediction['time_label']} ({prediction['time_confidence']:.1f}%)",
             tuple(self.config["color_time"])),
            (f"Weather: {prediction['weather_label']} ({prediction['weather_confidence']:.1f}%)",
             tuple(self.config["color_weather"])),
        ]

        y_offset = 20
        for text, color in texts:
            size = cv2.getTextSize(text, self.font, font_scale, thickness)[0]
            x = frame.shape[1] - size[0] - 20
            y = y_offset + size[1]
            cv2.putText(frame, text, (x, y), self.font, font_scale, color, thickness, line_type)
            y_offset += size[1] + 20

        return frame

    def show(self, frame, window_name="IP Camera"):
        cv2.imshow(window_name, frame)

    def record(self, frame):
        if self.out:
            resized = cv2.resize(frame, (self.config["width"], self.config["height"]))
            self.out.write(resized)

    def release(self):
        if self.out:
            self.out.release()

    def __del__(self):
        self.release()
