import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Ruta al conjunto de datos
data_dir = 'dataset'

# Parámetros del modelo
img_height, img_width = 150, 150
batch_size = 1  # Reducir el tamaño del lote
epochs = 10
learning_rate = 0.001

# Verificación y ajuste del dataset
def verify_dataset(data_dir):
    categories = ['Day', 'Evening', 'Night (Nightvision)', 'Clear', 'Cloudy', 'Partly_Cloudy']
    min_images = 5
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            if len(images) < min_images:
                raise ValueError(f"Category '{category}' has less than {min_images} images. Please ensure each category has at least {min_images} images.")

verify_dataset(data_dir)

# Preprocesamiento de datos
datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Mezclar las imágenes
)

# Verificación de la cantidad de imágenes encontradas
print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")

# Creación del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')  # Ajustar para 6 clases
])

# Compilación del modelo
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=epochs
)

# Guardar el modelo con la fecha actual
model_filename = f"day_evening_night_clear_cloudy_partly_cloudy_classifier_{datetime.now().strftime('%Y%m%d')}.h5"
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Visualización de los resultados
acc = history.history['accuracy']
loss = history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

def test_model_with_dataset(data_dir, model_filename):
    # Cargar el modelo entrenado
    model = load_model(model_filename)

    # Definir las etiquetas
    labels = ['Day', 'Evening', 'Night (Nightvision)', 'Clear', 'Cloudy', 'Partly_Cloudy']

    # Recorrer las imágenes del dataset
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                
                # Cargar y preprocesar la imagen
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Realizar la predicción
                predictions = model.predict(img_array)
                predicted_label = labels[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

                # Mostrar el resultado
                print(f"Image: {img_name}, Prediction: {predicted_label}, Confidence: {confidence:.2f}%")

# Prueba del modelo con las imágenes del dataset
test_model_with_dataset(data_dir, model_filename)
