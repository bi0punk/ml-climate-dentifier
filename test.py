import cv2
import os

RTSP_USER = os.getenv("RTSP_USER", "admin")
RTSP_PASS = os.getenv("RTSP_PASS", "")
RTSP_HOST = os.getenv("RTSP_HOST", "192.168.1.92")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
RTSP_PATH = os.getenv("RTSP_PATH", "/onvif1")
RTSP_URL = f'rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}:{RTSP_PORT}{RTSP_PATH}'

# Probar con 'tcp'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream with TCP')
    exit(-1)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture frame')
        break

    cv2.imshow('RTSP stream', frame)

    if cv2.waitKey(1) == 27:  # Esc key to stop
        break

cap.release()
cv2.destroyAllWindows()