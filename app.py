import logging
import sys
import argparse
import cv2

from inference.config import load_config
from inference.streamer import Streamer
from inference.predictor import Predictor
from inference.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


def parse_args():
    parser = argparse.ArgumentParser(description="IP Camera Weather and Time Classifier")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--record", action="store_true", help="Enable video recording")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.record:
        cfg["record"]["enabled"] = True

    logger.info("Starting IP Camera Weather Classifier")
    logger.info(f"Camera URL: {cfg['camera']['url']}")
    logger.info(f"Model: {cfg['model']['path']}")

    streamer = Streamer(cfg)
    if not streamer.open():
        logger.error("Could not open camera stream. Exiting.")
        sys.exit(1)

    predictor = Predictor(cfg)
    visualizer = Visualizer(cfg)

    last_prediction = {
        "time_label": "N/A",
        "time_confidence": 0.0,
        "weather_label": "N/A",
        "weather_confidence": 0.0,
    }

    logger.info("Running. Press 'q' to quit.")

    while True:
        ret, frame = streamer.read()
        if not ret:
            logger.warning("Failed to capture frame, reconnecting...")
            if not streamer.reconnect():
                logger.error("Could not reconnect. Exiting.")
                break
            continue

        if streamer.should_predict():
            prediction = predictor.predict(frame)
            last_prediction = prediction
            logger.debug(
                f"Time: {prediction['time_label']} ({prediction['time_confidence']:.1f}%) | "
                f"Weather: {prediction['weather_label']} ({prediction['weather_confidence']:.1f}%)"
            )
        else:
            prediction = last_prediction

        frame = visualizer.draw(frame, prediction)
        visualizer.show(frame)
        visualizer.record(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Quit signal received")
            break

    cv2.destroyAllWindows()
    streamer.release()
    visualizer.release()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
