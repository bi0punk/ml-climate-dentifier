#!/bin/bash

# Dirección del stream RTSP
RTSP_URL="rtsp://admin:pass@192.168.1.92:554/onvif1"

# Directorio donde se guardarán las imágenes
OUTPUT_DIR="./captures"

# Crear el directorio si no existe
mkdir -p "$OUTPUT_DIR"

# Intervalo de captura en segundos (15 minutos)
INTERVAL=300

# Bucle infinito para capturar imágenes cada 15 minutos
while true; do
    # Generar un nombre de archivo basado en el timestamp actual
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FILENAME="${OUTPUT_DIR}/capture_${TIMESTAMP}.jpg"

    # Capturar una imagen del stream y guardarla con el nombre generado
    ffmpeg -y -i "$RTSP_URL" -vframes 1 -q:v 2 "$FILENAME"

    echo "Imagen guardada como $FILENAME"

    # Esperar el intervalo antes de la próxima captura
    sleep "$INTERVAL"
done
