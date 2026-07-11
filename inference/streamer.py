import cv2
import time
import logging

logger = logging.getLogger(__name__)


class Streamer:
    def __init__(self, config):
        self.config = config["camera"]
        self.cap = None
        self.frame_count = 0
        self.frame_skip = config["inference"]["frame_skip"]
        self.reconnect_delay = self.config["reconnect_delay"]
        self.max_delay = self.config["max_reconnect_delay"]

    def open(self):
        url = self.config["url"]
        transport = self.config["transport"]
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"

        logger.info(f"Opening stream: {url}")
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap or not self.cap.isOpened():
            logger.error("Failed to open stream")
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config["buffer_size"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, self.config["fps"])
        self.reconnect_delay = self.config["reconnect_delay"]
        logger.info("Stream opened successfully")
        return True

    def read(self):
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        self.frame_count += 1
        return ret, frame

    def should_predict(self):
        return self.frame_count % self.frame_skip == 0

    def reconnect(self):
        delay = self.reconnect_delay
        logger.warning(f"Reconnecting in {delay:.1f}s...")
        time.sleep(delay)
        self.release()
        ok = self.open()
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_delay)
        return ok

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()
