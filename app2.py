import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

# Ruta al modelo entrenado
model_path = 'day_evening_night_weather_classifier_20240728.h5'

# Cargar el modelo
model = load_model(model_path)

# Definir las etiquetas
day_labels = ['Day', 'Evening', 'Night']
weather_labels = ['Clear', 'Cloudy', 'Partly_Cloudy']

# URL de la cámara IP
ip_camera_url = 'rtsp://admin:191448057devops@192.168.1.92:554/onvif1'

# Configurar rtsp_transport a 'tcp'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Intentar abrir la cámara IP
def open_video_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Ajustar el tamaño del buffer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error al abrir la cámara IP.")
        return None
    return cap

# Inicializar la captura de video
cap = open_video_capture(ip_camera_url)

# Dimensiones de la imagen
img_height, img_width = 150, 150

# Configuración de VideoWriter para guardar en H.265
fourcc = cv2.VideoWriter_fourcc(*'HEVC')
out = cv2.VideoWriter('output_h265.mp4', fourcc, 20.0, (640, 480))

def predict_frame(frame):
    # Preprocesar la imagen
    img = cv2.resize(frame, (img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Realizar la predicción
    day_prediction, weather_prediction = model.predict(img_array)
    predicted_day = day_labels[np.argmax(day_prediction)]
    predicted_weather = weather_labels[np.argmax(weather_prediction)]
    day_confidence = np.max(day_prediction) * 100
    weather_confidence = np.max(weather_prediction) * 100
    
    return predicted_day, day_confidence, predicted_weather, weather_confidence

while True:
    if cap is None:
        time.sleep(1)
        cap = open_video_capture(ip_camera_url)
        continue

    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        cap.release()
        cap = open_video_capture(ip_camera_url)
        continue
    
    predicted_day, day_confidence, predicted_weather, weather_confidence = predict_frame(frame)
    
    # Mostrar las etiquetas y la confianza en la imagen
    text_day = f"Day: {predicted_day} ({day_confidence:.2f}%)"
    text_weather = f"Weather: {predicted_weather} ({weather_confidence:.2f}%)"
    
    # Calcular la posición del texto en el lado derecho
    text_size_day = cv2.getTextSize(text_day, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_size_weather = cv2.getTextSize(text_weather, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x_day = frame.shape[1] - text_size_day[0] - 10
    text_x_weather = frame.shape[1] - text_size_weather[0] - 10
    text_y_day = text_size_day[1] + 10
    text_y_weather = text_size_day[1] + text_size_weather[1] + 20
    
    cv2.putText(frame, text_day, (text_x_day, text_y_day), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text_weather, (text_x_weather, text_y_weather), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Redimensionar la ventana
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
