import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

# Ruta al modelo entrenado
model_path = 'day_evening_night_clear_cloudy_partly_cloudy_classifier_20240923.h5'

# Cargar el modelo
model = load_model(model_path)

# Definir las etiquetas
time_of_day_labels = ['Day', 'Evening', 'Night']
weather_labels = ['Clear', 'Cloudy', 'Partly Cloudy']

# URL de la cámara IP
ip_camera_url = 'rtsp://admin:191448057devops@192.168.1.92:554/onvif1'



# Configurar rtsp_transport a 'tcp'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Intentar abrir la cámara IP
def open_video_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error al abrir la cámara IP.")
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Ajustar el tamaño del buffer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# Inicializar la captura de video
cap = open_video_capture(ip_camera_url)

# Dimensiones de la imagen
img_height, img_width = 150, 150

# Configuración de VideoWriter para guardar en un formato compatible
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes probar con 'MJPG' si 'XVID' no funciona
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

def predict_frame(frame):
    """ Realiza predicciones en el frame proporcionado. """
    img = cv2.resize(frame, (img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Realizar la predicción
    predictions = model.predict(img_array)[0]
    
    # Asumimos que las primeras 3 salidas son para tiempo del día y las últimas 3 para el clima
    time_of_day_pred = predictions[:3]   # Primeras 3 predicciones
    weather_pred = predictions[3:]       # Últimas 3 predicciones
    
    # Obtener las etiquetas y sus probabilidades
    time_of_day_label = time_of_day_labels[np.argmax(time_of_day_pred)]
    weather_label = weather_labels[np.argmax(weather_pred)]
    
    # Calcular la confianza
    time_of_day_confidence = np.max(time_of_day_pred) * 100
    weather_confidence = np.max(weather_pred) * 100
    
    return time_of_day_label, time_of_day_confidence, weather_label, weather_confidence

def draw_predictions(frame, time_of_day_label, time_of_day_confidence, weather_label, weather_confidence):
    """ Dibuja las predicciones en el frame. """
    # Configuración de fuentes y colores
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 255, 0)
    font_thickness = 3
    line_type = cv2.LINE_AA
    
    # Preparar el texto
    text_time_of_day = f"Time of Day: {time_of_day_label}: {time_of_day_confidence:.2f}%"
    text_weather = f"Weather: {weather_label}: {weather_confidence:.2f}%"
    
    # Calcular la posición del texto
    text_size_time_of_day = cv2.getTextSize(text_time_of_day, font, font_scale, font_thickness)[0]
    text_size_weather = cv2.getTextSize(text_weather, font, font_scale, font_thickness)[0]
    
    text_x_time_of_day = frame.shape[1] - text_size_time_of_day[0] - 20
    text_y_time_of_day = text_size_time_of_day[1] + 20
    
    text_x_weather = frame.shape[1] - text_size_weather[0] - 20
    text_y_weather = text_y_time_of_day + text_size_weather[1] + 20
    
    # Dibujar el texto en el frame
    cv2.putText(frame, text_time_of_day, (text_x_time_of_day, text_y_time_of_day), font, font_scale, font_color, font_thickness, line_type)
    cv2.putText(frame, text_weather, (text_x_weather, text_y_weather), font, font_scale, font_color, font_thickness, line_type)

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
    
    # Realizar la predicción en el frame
    time_of_day_label, time_of_day_confidence, weather_label, weather_confidence = predict_frame(frame)
    
    # Dibujar las predicciones en el frame
    draw_predictions(frame, time_of_day_label, time_of_day_confidence, weather_label, weather_confidence)
    
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
