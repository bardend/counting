import cv2
import os
from test_tensorflow_serving import predict
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)

os.makedirs("final", exist_ok=True)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

val = 5

i = 0
while True:
    # Leer un fotograma del video
    ret, frame = cap.read()

    # Verificar si el fotograma se leyó correctamente
    if not ret:
        print("No se pudo leer el fotograma")
        break

    if i % val == 0:
        vec = predict(frame)

        for r in vec:
            cv2.rectangle(frame, (r[1], r[2]), (r[3], r[4]), (255,0,0), 8)
            text = "Texto de ejemplo"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 255)  # Color del texto (blanco en este caso)
            thickness = 1  # Grosor del texto
            # Calcula la posición del texto para que esté justo arriba del rectángulo
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = r[1] + (r[3] - r[1]) // 2 - text_size[0] // 2
            text_y = r[2] - 10  # Desplaza el texto 10 píxeles por encima del rectángulo
            cv2.putText(frame, r[0], (text_x, text_y), font, font_scale, color, thickness)


        filename = f"img_{i}.jpg"
        output_path = os.path.join("final", filename)

        cv2.imwrite(output_path, frame)
        print(len(vec))

        

    # Mostrar el fotograma en una ventana
    cv2.imshow('Video en tiempo real', frame)

    # Esperar 1 milisegundo y verificar si se presionó la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

