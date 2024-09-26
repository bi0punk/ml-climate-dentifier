import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

# Ruta al modelo entrenado
MODEL_PATH = 'day_evening_night_clear_cloudy_partly_cloudy_classifier_20240923.h5'

# Cargar el modelo
model = load_model(MODEL_PATH)

# Definir las etiquetas
TIME_OF_DAY_LABELS = ['Day', 'Evening', 'Night']
WEATHER_LABELS = ['Clear', 'Cloudy', 'Partly Cloudy']

# URL de la cámara IP
IP_CAMERA_URL = 'rtsp://admin:191448057devops@192.168.1.92:554/onvif1'

# Configurar rtsp_transport a 'tcp'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Intentar abrir la cámara IP
def open_video_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error al abrir la cámara IP.")
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# Inicializar la captura de video
cap = open_video_capture(IP_CAMERA_URL)

# Dimensiones de la imagen
IMG_HEIGHT, IMG_WIDTH = 150, 150

# Configuración de VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

def predict_frame(frame):
    """Realiza predicciones en el frame proporcionado."""
    img = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)[0]
    
    # Obtener predicciones
    time_of_day_pred = predictions[:3]
    weather_pred = predictions[3:]
    
    time_of_day_label = TIME_OF_DAY_LABELS[np.argmax(time_of_day_pred)]
    weather_label = WEATHER_LABELS[np.argmax(weather_pred)]
    
    time_of_day_confidence = np.max(time_of_day_pred) * 100
    weather_confidence = np.max(weather_pred) * 100
    
    return time_of_day_label, time_of_day_confidence, weather_label, weather_confidence

def draw_predictions(frame, time_of_day_label, time_of_day_confidence, weather_label, weather_confidence):
    """Dibuja las predicciones en el frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 255, 0)
    font_thickness = 3
    line_type = cv2.LINE_AA
    
    text_time_of_day = f"Time of Day: {time_of_day_label}: {time_of_day_confidence:.2f}%"
    text_weather = f"Weather: {weather_label}: {weather_confidence:.2f}%"
    
    # Calcular la posición del texto
    y_offset = 20
    for text in [text_time_of_day, text_weather]:
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = y_offset + text_size[1]
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, line_type)
        y_offset += text_size[1] + 20  # Espaciado entre líneas

while True:
    if cap is None:
        time.sleep(1)
        cap = open_video_capture(IP_CAMERA_URL)
        continue

    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        cap.release()
        cap = open_video_capture(IP_CAMERA_URL)
        continue
    
    # Realizar la predicción en el frame
    time_of_day_label, time_of_day_confidence, weather_label, weather_confidence = predict_frame(frame)
    
    # Dibujar las predicciones en el frame
    draw_predictions(frame, time_of_day_label, time_of_day_confidence, weather_label, weather_confidence)
    
    # Redimensionar la ventana para mostrar
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Mostrar la imagen
    cv2.imshow('IP Camera', frame_resized)
    
    # Guardar el frame en el archivo de video
    out.write(frame_resized)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
