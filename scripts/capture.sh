#!/bin/bash
# Periodically capture frames from IP camera for dataset building
# Usage: ./capture.sh [interval_seconds]
# Default interval: 180 seconds (3 minutes)

INTERVAL=${1:-180}
RTSP_URL="${RTSP_URL:-http://192.168.1.82:81/stream}"
OUTPUT_DIR="./captures"

if [ -z "$RTSP_URL" ]; then
    echo "Error: RTSP_URL not set. Use env var or edit script."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Capturing from: $RTSP_URL"
echo "Interval: ${INTERVAL}s"
echo "Output: $OUTPUT_DIR"
echo "Press Ctrl+C to stop."

while true; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FILENAME="${OUTPUT_DIR}/capture_${TIMESTAMP}.jpg"
    ffmpeg -y -i "$RTSP_URL" -vframes 1 -q:v 2 "$FILENAME" 2>/dev/null
    echo "Saved: $FILENAME ($(date))"
    sleep "$INTERVAL"
done
