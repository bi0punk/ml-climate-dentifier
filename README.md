rea# IP Camera Weather and Time Classifier

Este proyecto utiliza una cámara IP para capturar y clasificar imágenes en función del tiempo del día y las condiciones climáticas. El modelo de clasificación fue entrenado utilizando TensorFlow y Keras.

## Requisitos

- Python 3.7+
- OpenCV
- TensorFlow
- NumPy

## Instalación

1. Clona este repositorio:
    ```sh
    git clone https://github.com/tu_usuario/ip-camera-weather-classifier.git
    cd ip-camera-weather-classifier
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Asegúrate de que el modelo entrenado `day_evening_night_clear_cloudy_partly_cloudy_classifier_20240729.h5` esté en el directorio del proyecto.

2. Modifica la URL de la cámara IP en el script si es necesario:
    ```python
    ip_camera_url = 'rtsp://admin:pass@ip:554/onvif1'
    ```

3. Ejecuta el script:
    ```sh
    python main.py
    ```

## Estructura del proyecto

```plaintext
.
├── README.md
├── main.py
├── day_evening_night_clear_cloudy_partly_cloudy_classifier_20240729.h5
└── requirements.txt
