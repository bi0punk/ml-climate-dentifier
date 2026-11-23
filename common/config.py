"""Carga compartida de configuración (config.yaml) y variables de entorno."""

import os
from urllib.parse import quote

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _camera_url_from_env():
    """Construye la URL de cámara desde variables de entorno.

    Prioridad: ``IP_CAMERA_URL`` > ``RTSP_URL``. Si la URL no trae credenciales y
    existen ``RTSP_USER``/``RTSP_PASSWORD``, se insertan en la URL.
    """
    url = os.getenv("IP_CAMERA_URL") or os.getenv("RTSP_URL")
    if not url:
        return None

    url = os.path.expandvars(url)
    user = os.getenv("RTSP_USER", "")
    password = os.getenv("RTSP_PASSWORD", "")
    if not os.getenv("IP_CAMERA_URL") and user and password and "@" not in url.split("://", 1)[-1]:
        scheme, separator, rest = url.partition("://")
        if separator:
            url = f"{scheme}://{quote(user, safe='')}:{quote(password, safe='')}@{rest}"
    return url


def load_config(config_path=None):
    """Carga config.yaml y aplica overrides por variables de entorno.

    - ``IP_CAMERA_URL`` o ``RTSP_URL`` sobreescriben ``camera.url``
    - ``RTSP_USER``/``RTSP_PASSWORD`` completan credenciales si la URL no las trae
    - ``MODEL_PATH`` sobreescribe ``model.path``
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    camera = cfg.setdefault("camera", {})
    camera["url"] = os.path.expandvars(str(camera.get("url", "")))

    env_url = _camera_url_from_env()
    if env_url:
        camera["url"] = env_url

    env_model = os.getenv("MODEL_PATH")
    if env_model:
        cfg["model"]["path"] = env_model

    return cfg