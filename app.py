import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Ruta al modelo entrenado
model_path = 'day_evening_night_classifier.h5'

# Cargar el modelo
model = load_model(model_path)

# Definir las etiquetas
labels = ['Day', 'Evening', 'Night']

# URL de la cámara IP
ip_camera_url = 'rtsp://admin:191448057devops@192.168.1.92:554/onvif1'

# Inicializar la captura de video
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error al abrir la cámara IP.")
    exit()

# Dimensiones de la imagen
img_height, img_width = 150, 150

def predict_frame(frame):
    # Preprocesar la imagen
    img = cv2.resize(frame, (img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return predicted_label, confidence

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break
    
    predicted_label, confidence = predict_frame(frame)
    
    # Mostrar la etiqueta y la confianza en la imagen
    text = f"{predicted_label}: {confidence:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostrar la imagen
    cv2.imshow('IP Camera', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
