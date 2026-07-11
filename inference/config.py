import os
import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(config_path=None):
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env_url = os.getenv("IP_CAMERA_URL")
    if env_url:
        cfg["camera"]["url"] = env_url

    env_model = os.getenv("MODEL_PATH")
    if env_model:
        cfg["model"]["path"] = env_model

    return cfg
