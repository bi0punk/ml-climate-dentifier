# 🌤️ ML Climate Dentifier

**IP Camera Weather and Time-of-Day Classifier** — clasifica en tiempo real el momento del día (Day/Evening/Night) y el clima (Clear/Cloudy/Partly Cloudy) desde una cámara IP usando una CNN de doble cabeza con TensorFlow/Keras.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python)](https://python.org)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![OpenCV 4.10](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/bi0punk/ml-climate-dentifier)](https://github.com/bi0punk/ml-climate-dentifier/commits/main)

---

## 📑 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Arquitectura](#-arquitectura)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Dataset](#-dataset)
- [Configuración](#-configuración)
- [Uso: Inferencia en Vivo](#-uso-inferencia-en-vivo)
- [Uso: Entrenamiento](#-uso-entrenamiento)
- [Uso: Scripts](#-uso-scripts)
- [Modelos](#-modelos)
- [Docker](#-docker)
- [Tests](#-tests)
- [Rendimiento Esperado](#-rendimiento-esperado)
- [Troubleshooting](#-troubleshooting)
- [v1 → v2: Migración](#-v1--v2-migración)
- [Roadmap](#-roadmap)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

---

## 📖 Descripción General

Este proyecto nació como un experimento de computer vision: capturar el stream de una cámara IP exterior y clasificar automáticamente **dos atributos independientes** de la escena:

| Atributo | Clases |
|----------|--------|
| **Momento del día** (`Time of Day`) | Day, Evening, Night |
| **Clima** (`Weather`) | Clear, Cloudy, Partly Cloudy |

**Caso de uso típico**: Cámara IP apuntando al exterior (jardín, calle, paisaje). El sistema captura frames en vivo, los pasa por una CNN y overlay los resultados en el video con porcentajes de confianza.

**Dataset actual**: ~1,430 imágenes etiquetadas manualmente, capturadas entre Jul-Sep 2024, con desbalance significativo entre clases (ver sección Dataset).

---

## 🏗️ Arquitectura

### El Problema en v1 (solucionado)

La versión 1 entrenaba un modelo con **una sola softmax de 6 clases** donde `Day`, `Evening`, `Night`, `Clear`, `Cloudy` y `Partly Cloudy` **competían entre sí**. Esto no tiene sentido semántico: el momento del día y el clima son atributos independientes. Una imagen puede ser "Day" y "Clear" simultáneamente. Además, `app.py` dividía artificialmente las 6 salidas en dos grupos de 3, pero el modelo nunca fue entrenado para producir distribuciones separadas.

### Solución: Dual-Head Architecture (v2)

El modelo ahora tiene **dos cabezas de salida independientes**, cada una con su propia softmax:

```
Input (150x150x3)
  │
  ├── Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
  ├── Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
  ├── Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
  ├── Conv2D(256, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
  │
  ├── Flatten → Dense(256, ReLU) → Dropout(0.5)
  │           → Dense(128, ReLU) → Dropout(0.3)
  │
  ├── [time_head]    Dense(3, softmax)     ← Day / Evening / Night
  └── [weather_head] Dense(3, softmax)     ← Clear / Cloudy / Partly Cloudy
```

**Pérdida combinada**:
```
Loss = categorical_crossentropy(time_labels, time_pred)
     + categorical_crossentropy(weather_labels, weather_pred)
```

### Sample-Weight Masking

Para entrenar con datos parcialmente etiquetados (imágenes que solo tienen label de tiempo O de clima), se usa **sample-weight masking**:

- Imágenes de `dataset/day/`: `weight_time=1.0`, `weight_weather=0.0` → solo contribuyen a la pérdida de tiempo
- Imágenes de `dataset/clear/`: `weight_time=0.0`, `weight_weather=1.0` → solo contribuyen a la pérdida de clima

Esto permite usar **todas las imágenes disponibles** sin necesidad de tener ambas etiquetas.

### Backbone Alternativo: MobileNetV2

```bash
python -m training.train --backbone mobilenetv2
```

Usa MobileNetV2 pre-entrenado en ImageNet como feature extractor (convolutional base congelada), ideal cuando el dataset es pequeño y se necesita mejor generalización.

---

## 📁 Estructura del Proyecto

```
ml-climate-dentifier/
│
├── app.py                      # Entry point: clasificación en tiempo real
├── config.yaml                 # Configuración centralizada (YAML)
├── requirements.txt            # Dependencias Python
├── pyproject.toml              # Metadata del proyecto + pytest config
├── Dockerfile                  # Imagen Docker lista para deploy
├── Makefile                    # Comandos comunes (run, train, test, docker)
│
├── inference/                  # Paquete de inferencia en vivo
│   ├── __init__.py
│   ├── config.py               # Carga YAML + overrides de .env
│   ├── predictor.py            # Carga del modelo, predict, smoothing temporal
│   ├── streamer.py             # Captura de cámara con reconexión exponencial
│   └── visualizer.py           # Overlay OpenCV, display, grabación
│
├── training/                   # Paquete de entrenamiento
│   ├── __init__.py
│   ├── model.py                # Arquitectura CNN / MobileNetV2 dual-head
│   ├── dataset.py              # DualDataGenerator con sample-weight masking
│   ├── train.py                # Loop de entrenamiento con callbacks
│   └── evaluate.py             # Matrices de confusión, classification reports
│
├── scripts/                    # Utilidades
│   ├── build_dataset.py        # Combina directorios time+weather en 9 clases
│   ├── label_captures.py       # Etiqueta captures/ por timestamp
│   └── capture.sh              # Captura periódica de frames via ffmpeg
│
├── tests/                      # Suite de tests (pytest)
│   ├── __init__.py
│   ├── test_config.py          # Tests del cargador de configuración
│   ├── test_dataset.py         # Tests del DualDataGenerator
│   ├── test_model.py           # Tests de la arquitectura (shapes, outputs)
│   └── test_predictor.py       # Tests del predictor y smoothing
│
├── models/                     # Modelos .h5 entrenados (gitignored)
├── dataset/                    # Dataset RAW (gitignored)
│   ├── day/                    #   628 imágenes — Day (time)
│   ├── evening/                #   132 imágenes — Evening (time)
│   ├── night (Nightvision)/    #   585 imágenes — Night (time)
│   ├── clear/                  #    65 imágenes — Clear (weather)
│   ├── cloudy/                 #    10 imágenes — Cloudy (weather)
│   └── partly_cloudy/          #    10 imágenes — Partly Cloudy (weather)
├── captures/                   # Capturas periódicas (gitignored)
│
├── .env.example                # Template de variables de entorno
├── .gitignore
└── README.md
```

---

## 🔧 Instalación

### Requisitos

- Python 3.9+
- pip
- OpenCV (se instala vía pip, pero necesita `libgl1` en Linux)
- Opcional: GPU con CUDA para entrenar más rápido

### Pasos

```bash
# 1. Clonar
git clone https://github.com/bi0punk/ml-climate-dentifier
cd ml-climate-dentifier

# 2. (Recomendado) Crear virtualenv
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar entorno
cp .env.example .env
# Editar .env con URL de tu cámara y ruta del modelo

# 5. Verificar instalación
python -c "import cv2, tensorflow, numpy; print('OK')"
```

### Dependencias

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `tensorflow` | 2.17+ | Entrenamiento e inferencia del modelo |
| `opencv-python` | 4.10+ | Captura de video, procesamiento de frames, visualización |
| `numpy` | 2.0+ | Operaciones numéricas y preprocesamiento |
| `matplotlib` | 3.9+ | Gráficas de entrenamiento |
| `scikit-learn` | 1.5+ | Métricas de evaluación (confusion matrix, classification report) |
| `python-dotenv` | 1.0+ | Carga de variables de entorno desde `.env` |
| `pyyaml` | 6.0+ | Parseo de `config.yaml` |

---

## 📊 Dataset

### Estructura Actual (RAW)

| Directorio | Cantidad | Tipo de Label | Desbalance vs Clase Mayoritaria |
|------------|----------|---------------|---------------------------------|
| `day/` | 628 | Time: Day | — (mayoría) |
| `evening/` | 132 | Time: Evening | 4.8× menos que Day |
| `night (Nightvision)/` | 585 | Time: Night | 1.1× menos que Day |
| `clear/` | 65 | Weather: Clear | — (mayoría weather) |
| `cloudy/` | 10 | Weather: Cloudy | 6.5× menos que Clear |
| `partly_cloudy/` | 10 | Weather: Partly Cloudy | 6.5× menos que Clear |
| **Total** | **1,430** | — | — |

### Análisis de Desbalance

El dataset tiene **dos problemas de desbalance**:

1. **Time**: Evening (132) está severamente subrepresentado vs Day (628) y Night (585). Las imágenes de Evening se capturaron en un solo día (2024-07-29).
2. **Weather**: Cloudy y Partly Cloudy (10 c/u) tienen 6.5× menos imágenes que Clear (65). No hay suficientes ejemplos para que el modelo aprenda robustez climática.

### Cómo se usan las etiquetas parciales

Cada imagen tiene solo **un label conocido** (time O weather, no ambos). El `DualDataGenerator` asigna:

- Para imágenes de `day/`, `evening/`, `night/`:
  - `time_label` = one-hot conocido
  - `weather_label` = [0, 0, 0] (desconocido)
  - `sample_weight` = `[1.0, 0.0]` → solo contribuye a la pérdida de tiempo

- Para imágenes de `clear/`, `cloudy/`, `partly_cloudy/`:
  - `time_label` = [0, 0, 0] (desconocido)
  - `weather_label` = one-hot conocido
  - `sample_weight` = `[0.0, 1.0]` → solo contribuye a la pérdida de clima

### Script: build_dataset.py

Para entrenar modelos de 9 clases combinadas (alternativa al dual-head):

```bash
# Ver distribución actual
python scripts/build_dataset.py --analyze

# Construir dataset combinado (usa symlinks por defecto)
python scripts/build_dataset.py

# Para copiar en vez de symlinks
python scripts/build_dataset.py --copy
```

### Script: label_captures.py

Las imágenes en `captures/` tienen timestamps en el nombre (`capture_20241015_151234.jpg`). Este script infiere el momento del día:

```bash
python scripts/label_captures.py
```

Lógica de inferencia:
- 06:00–18:00 → `day/`
- 18:00–21:00 → `evening/`
- 21:00–06:00 → `night/`

---

## ⚙️ Configuración

### config.yaml (Referencia Completa)

| Ruta | Campo | Tipo | Default | Descripción |
|------|-------|------|---------|-------------|
| `camera.url` | string | `"http://192.168.1.82:81/stream"` | URL del stream (HTTP, RTSP, MJPEG) |
| `camera.width` | int | `640` | Ancho de captura |
| `camera.height` | int | `480` | Alto de captura |
| `camera.fps` | int | `30` | FPS objetivo de captura |
| `camera.buffer_size` | int | `10` | Tamaño del buffer OpenCV |
| `camera.transport` | string | `"udp"` | Transporte RTSP (`udp` o `tcp`) |
| `camera.reconnect_delay` | float | `1.0` | Espera inicial antes de reconectar (s) |
| `camera.max_reconnect_delay` | float | `60.0` | Espera máxima tras fallos repetidos (s) |
| `model.path` | string | `"models/..."` | Ruta al archivo `.h5` |
| `model.input_size` | int | `150` | Tamaño esperado por el modelo (px) |
| `model.type` | string | `"single_6class"` | Formato: `single_6class` o `dual_head` |
| `inference.frame_skip` | int | `5` | Predecir cada N frames |
| `inference.smoothing_window` | int | `5` | Ventana para filtro de moda |
| `inference.confidence_threshold` | float | `0.0` | Confianza mínima para mostrar |
| `display.width` | int | `640` | Ancho de ventana de visualización |
| `display.height` | int | `480` | Alto de ventana de visualización |
| `display.font_scale` | float | `1.5` | Tamaño de fuente del overlay |
| `display.font_thickness` | int | `3` | Grosor de fuente |
| `display.color_time` | list | `[0, 255, 0]` | Color BGR del texto de tiempo |
| `display.color_weather` | list | `[0, 255, 0]` | Color BGR del texto de clima |
| `record.enabled` | bool | `false` | Grabar video a archivo |
| `record.output_path` | string | `"output.avi"` | Ruta del video de salida |
| `record.codec` | string | `"XVID"` | Códec de video |
| `record.fps` | float | `20.0` | FPS del video grabado |
| `labels.time` | list | `["Day", "Evening", "Night"]` | Nombres de clases de tiempo |
| `labels.weather` | list | `["Clear", "Cloudy", "Partly Cloudy"]` | Nombres de clases de clima |

### .env (Variables de Entorno)

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `IP_CAMERA_URL` | URL del stream (sobreescribe config.yaml) | `rtsp://192.168.1.92:554/onvif1` |
| `MODEL_PATH` | Ruta al modelo (sobreescribe config.yaml) | `models/mi_modelo.h5` |
| `RTSP_URL` | URL completa con credenciales | `rtsp://user:pass@ip:554/stream` |

Las variables de entorno **siempre tienen prioridad** sobre `config.yaml`.

---

## 🎥 Uso: Inferencia en Vivo

### Comandos Básicos

```bash
# Usando config.yaml por defecto
python app.py

# Con grabación de video activada
python app.py --record

# Con archivo de configuración personalizado
python app.py --config mi_config.yaml
```

### Controles

| Tecla | Acción |
|-------|--------|
| `q` | Salir de la aplicación |

### Cómo Funciona el Pipeline de Inferencia

```
Cámara IP (HTTP/RTSP)
  │
  ▼
Streamer.read() → frame
  │
  ▼
[Frame Skip] ¿frame_count % frame_skip == 0?
  │                     │
  Sí                    No
  ▼                     ▼
Predictor.predict()    Usar última predicción (sin recalcular)
  │
  ▼
Smoothing temporal (moda de últimas N predicciones)
  │
  ▼
Visualizer.draw() → overlay de texto en el frame
  │
  ▼
cv2.imshow() + (opcional) VideoWriter.write()
```

**Frame Skip**: Predecir en **cada frame** es innecesario (el clima no cambia 30 veces por segundo). Con `frame_skip=5`, si la cámara entrega 30fps, solo se predice ~6fps, reduciendo el uso de CPU/GPU 5×.

**Smoothing Temporal**: Las predicciones individuales pueden ser ruidosas. Se aplica un filtro de **moda** sobre una ventana deslizante de las últimas N predicciones, estabilizando el resultado.

**Reconexión Automática**: Si el stream se cae, el `Streamer` intenta reconectar con **exponential backoff**: 1s → 2s → 4s → 8s ... hasta 60s máximo. Al recuperarse, retoma normalmente.

### Model Type: single_6class vs dual_head

| Tipo | Descripción | Cuándo usarlo |
|------|-------------|---------------|
| `single_6class` | Una softmax de 6 salidas, `app.py` divide `[:3]` (time) y `[3:]` (weather) | Modelos v1 (entrenados con `trainer.py` original) |
| `dual_head` | Dos softmax independientes de 3 salidas cada una | Modelos v2 (entrenados con `training.train` nuevo) |

---

## 🧠 Uso: Entrenamiento

### Comandos

```bash
# Entrenamiento básico (CNN propia)
python -m training.train

# Especificar número de épocas y batch size
python -m training.train --epochs 50 --batch-size 64

# Transfer Learning con MobileNetV2
python -m training.train --backbone mobilenetv2

# Entrenar y evaluar al finalizar
python -m training.train --evaluate

# Dataset en otra ruta
python -m training.train --data-dir data/processed

# Guardar modelos en otra carpeta
python -m training.train --model-dir mi_carpeta
```

### Parámetros

| Flag | Default | Descripción |
|------|---------|-------------|
| `--data-dir` | `dataset` | Ruta al dataset raw |
| `--model-dir` | `models` | Carpeta para guardar modelos |
| `--backbone` | `cnn` | Arquitectura: `cnn` o `mobilenetv2` |
| `--batch-size` | `32` | Tamaño de lote |
| `--epochs` | `30` | Épocas máximas (early stopping puede parar antes) |
| `--learning-rate` | `0.001` | Tasa de aprendizaje inicial |
| `--patience` | `7` | Épocas sin mejora para early stopping |
| `--evaluate` | `false` | Ejecutar evaluación al finalizar |

### Callbacks Incluidos

| Callback | Monitorea | Comportamiento |
|----------|-----------|----------------|
| **EarlyStopping** | `val_loss` | Detiene el entrenamiento si no mejora tras 7 épocas; restaura los mejores pesos |
| **ModelCheckpoint** | `val_loss` | Guarda el mejor modelo observado durante el entrenamiento |
| **ReduceLROnPlateau** | `val_loss` | Reduce la LR a la mitad si no hay mejora tras 3 épocas, mínimo `1e-6` |
| **CSVLogger** | — | Guarda todas las métricas por época en `training_log_TIMESTAMP.csv` |

### Output del Entrenamiento

```
models/
├── best_model_20240710_120000.h5       # Mejor modelo (val_loss mínimo)
├── weather_classifier_20240710_120000.h5  # Modelo final (última época)
├── training_log_20240710_120000.csv      # Métricas por época
├── training_plot_20240710_120000.png     # Gráficas de accuracy/loss
└── config_20240710_120000.json           # Configuración usada
```

### Evaluación

Si se usa `--evaluate`, al terminar el entrenamiento se ejecuta:

1. **Matriz de confusión** para tiempo y clima (gráfica)
2. **Classification report** con precision, recall, f1-score por clase
3. Resultados en consola + gráficos guardados

---

## 📜 Uso: Scripts

### build_dataset.py

Construye un dataset de 9 clases combinadas (`day_clear`, `evening_cloudy`, `night_partly_cloudy`, etc.) a partir de los directorios separados. Útil si quieres entrenar un modelo de 9 clases en vez del dual-head.

```bash
# Análisis sin modificar archivos
python scripts/build_dataset.py --analyze

# Construir dataset combinado en data/processed/
python scripts/build_dataset.py

# Copiar imágenes (default: symlinks)
python scripts/build_dataset.py --copy

# Directorios personalizados
python scripts/build_dataset.py --raw-dir dataset --output-dir data/processed
```

### label_captures.py

Procesa las imágenes en `captures/` y las organiza por momento del día según el timestamp en el nombre del archivo.

```bash
python scripts/label_captures.py
# -> Organiza captures/ en data/captures_labeled/{day,evening,night}/
```

### capture.sh

Captura frames periódicamente de la cámara IP usando ffmpeg.

```bash
# Cada 3 minutos (default)
./scripts/capture.sh

# Cada 10 minutos
./scripts/capture.sh 600

# Con variable de entorno
RTSP_URL="rtsp://..." ./scripts/capture.sh
```

---

## 🤖 Modelos

### Modelos Disponibles

| Archivo | Fecha | Clases | Tipo | Descripción |
|---------|-------|--------|------|-------------|
| `day_evening_night_classifier_20240728.h5` | 2024-07-28 | 3 (time only) | v1 single_6class | Solo clasifica momento del día |
| `day_evening_night_clear_cloudy_partly_cloudy_classifier_20240729.h5` | 2024-07-29 | 6 (time+weather) | v1 single_6class | Primer intento de clasificación conjunta |
| `day_evening_night_clear_cloudy_partly_cloudy_classifier_20240923.h5` | 2024-09-23 | 6 (time+weather) | v1 single_6class | Modelo v1 mejorado (usado por `app.py` por defecto) |

### Model Type Compatibility

| Modelo | `config.yaml model.type` | Funciona con |
|--------|--------------------------|--------------|
| v1 (6 clases, softmax única) | `single_6class` | `app.py`, `test.py` |
| v2 (dual-head, 2 salidas) | `dual_head` | `app.py`, `training/` |

Para usar un modelo v2, cambia en `config.yaml`:
```yaml
model:
  path: "models/weather_classifier_20240710.h5"
  type: "dual_head"
```

---

## 🐳 Docker

### Build

```bash
make docker-build
# o manual:
docker build -t ml-climate-dentifier .
```

### Run

```bash
make docker-run
# o manual (con cámara USB local):
docker run --rm \
  --device=/dev/video0:/dev/video0 \
  -e IP_CAMERA_URL="rtsp://..." \
  ml-climate-dentifier
```

La imagen usa `python:3.10-slim` e incluye OpenCV con soporte ffmpeg.

---

## 🧪 Tests

```bash
# Ejecutar toda la suite
make test

# O directamente con pytest
python -m pytest tests/ -v

# Con cobertura
pip install pytest-cov
python -m pytest tests/ --cov=inference --cov=training
```

### Tests Incluidos

| Archivo | Lo que prueba |
|---------|---------------|
| `test_config.py` | Carga de config.yaml, override de env vars, configs personalizados |
| `test_predictor.py` | Inicialización del predictor, lógica de smoothing |
| `test_dataset.py` | one-hot encoding, carga del dataset, shapes del generador |
| `test_model.py` | Creación de modelo CNN/MobileNetV2, shapes de salida, sumas de softmax |

---

## ⚡ Rendimiento Esperado

| Escenario | Backbone | Frame Skip | FPS Aprox | Plataforma |
|-----------|----------|------------|-----------|------------|
| Inferencia CPU | CNN | 5 | ~10-15 fps | Laptop moderna (i7, 16GB) |
| Inferencia CPU | MobileNetV2 | 5 | ~8-12 fps | Laptop moderna (i7, 16GB) |
| Inferencia GPU | CNN | 5 | ~25-30 fps | NVIDIA GTX 1660+ |
| Inferencia GPU | MobileNetV2 | 5 | ~25-30 fps | NVIDIA GTX 1660+ |
| Entrenamiento | CNN | — | ~5 min/30 epochs | NVIDIA GTX 1660 |
| Entrenamiento | MobileNetV2 | — | ~8 min/30 epochs | NVIDIA GTX 1660 |

*Nota: Los FPS dependen de la cámara, red, y resolución. Los tiempos de entrenamiento dependen del tamaño del dataset.*

---

## 🔧 Troubleshooting

### Error: `model.summary()` falla o `Unknown model type`

**Causa**: El `model.type` en `config.yaml` no coincide con el formato del archivo `.h5`.

**Solución**: Verifica qué tipo de modelo tienes:
```bash
python -c "
from tensorflow.keras.models import load_model
m = load_model('models/tu_modelo.h5')
print('Outputs:', len(m.outputs))
for o in m.outputs:
    print(' -', o.name, o.shape)
"
```
- Si 1 output → `single_6class`
- Si 2 outputs → `dual_head`

### Error: `Cannot open stream`

**Causas posibles**: URL incorrecta, cámara apagada, firewall, credenciales inválidas.

**Soluciones**:
1. Verificar la URL en el navegador: debe mostrar el stream
2. Probar con `test.py` (usa `.env`)
3. Cambiar `transport` de `udp` a `tcp` en `config.yaml`
4. Aumentar `buffer_size`
5. Verificar que la cámara soporte el formato solicitado

### Error: Predicciones siempre con confianza baja (< 50%)

**Causas**: Dataset desbalanceado, modelo mal entrenado, distribución diferente entre train y producción.

**Soluciones**:
1. Reentrenar con más datos de la clase débil
2. Usar `--backbone mobilenetv2`
3. Aumentar `--epochs` y reducir `--learning-rate`
4. Recopilar más imágenes del escenario real donde se desplegará

### Error: `CUDA out of memory`

**Solución**: Reducir `--batch-size` durante entrenamiento, o usar CPU:
```bash
CUDA_VISIBLE_DEVICES="" python -m training.train --batch-size 16
```

### Error: `FFmpeg not found`

En Docker está incluido. En la máquina host:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## 📋 v1 → v2: Migración

| Aspecto | v1 | v2 |
|---------|----|----|
| **Arquitectura** | Una softmax 6-way (Day, Evening, Night, Clear, Cloudy, Partly Cloudy compiten) | Dual-head: dos softmax independientes (3 time + 3 weather) |
| **Entrenamiento** | Sin validation split, `batch_size=1`, código de aumentación muerto | Train/val/test split, `batch_size=32`, aumentación activa, callbacks |
| **Inferencia** | Predice cada frame sin skip, sin suavizado | Frame skipping + filtro de moda temporal |
| **Configuración** | Hardcodeada en el código fuente | `config.yaml` + overrides via `.env` |
| **Seguridad** | Credenciales harcodeadas en `test.py` | Solo via `.env`, `.env.example` documentado |
| **Dataset** | Sin manejo de labels parciales | Sample-weight masking para etiquetas parciales |
| **Modelos** | Un solo archivo suelto en la raíz | Organizados en `models/` |
| **Estructura** | Archivos monolíticos | Paquetes modulares (`inference/`, `training/`, `scripts/`, `tests/`) |
| **Tests** | Ninguno | Suite pytest (config, predictor, dataset, modelo) |
| **Docker** | Ninguno | Dockerfile + Makefile |
| **Logging** | `print()` | `logging` estructurado |

---

## 🗺️ Roadmap

### Corto Plazo
- [ ] Balancear el dataset (más imágenes evening, cloudy, partly_cloudy)
- [ ] Entrenar y publicar modelo v2 con métricas de evaluación
- [ ] Agregar detección de lluvia (rain class)
- [ ] Dashboard web con Flask/FastAPI para mostrar histórico

### Mediano Plazo
- [ ] Interfaz web con gráficas de tendencias (evolución del clima en el tiempo)
- [ ] Almacenamiento de clasificaciones en SQLite/PostgreSQL
- [ ] Soporte para múltiples cámaras simultáneas
- [ ] Sistema de alertas (cambio brusco de clima, detección de eventos)

### Largo Plazo
- [ ] Despliegue en edge devices (Raspberry Pi, Jetson Nano)
- [ ] Modelo de segmentación semántica para análisis más granular
- [ ] Integración con home automation (Home Assistant, OpenHAB)
- [ ] Aplicación móvil para monitoreo remoto

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Haz tus cambios
4. Ejecuta los tests: `make test`
5. Commit: `git commit -m "feat: descripción clara"`
6. Push: `git push origin feature/nueva-funcionalidad`
7. Abre un Pull Request

### Convenciones de Código
- Sigue el estilo existente (mira archivos vecinos antes de editar)
- Usa type hints en funciones nuevas
- Agrega docstrings (estilo Google) en módulos y clases nuevas
- Los tests son obligatorios para funcionalidad nueva
- Commits en inglés, descriptivos, con prefijo (`feat:`, `fix:`, `docs:`, `chore:`)

---

## 📄 Licencia

MIT © 2024 bi0punk

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
