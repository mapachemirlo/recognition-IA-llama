import cv2
import torch
from random import choice

# Cargar el modelo preentrenado de YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Lista de objetos para desafíos (tomados del modelo YOLOv5 preentrenado)
object_classes = model.names  # Lista de clases que reconoce YOLOv5
challenge_objects = ["person", "car", "apple", "cat", "dog", "book", "chair"]

# Crear un desafío aleatorio
def generate_challenge():
    return choice(challenge_objects)

# Función principal del juego
def play_game():
    print("¡Bienvenido al juego educativo de reconocimiento de objetos!")
    print("Prepara la cámara...")

    # Abrir la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Crear un desafío
    current_challenge = generate_challenge()
    print(f"Encuentra algo que sea: {current_challenge.upper()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el cuadro.")
            break

        # Realizar predicciones
        results = model(frame)

        # Renderizar las predicciones en el cuadro
        result_frame = results.render()[0]
        detections = results.pandas().xyxy[0]  # Obtiene los resultados como DataFrame

        # Verificar si el objeto del desafío está en las detecciones
        detected_objects = detections['name'].tolist()
        if current_challenge in detected_objects:
            print(f"¡Bien hecho! Encontraste un(a): {current_challenge.upper()}")
            current_challenge = generate_challenge()  # Generar un nuevo desafío
            print(f"Tu próximo desafío es: {current_challenge.upper()}")

        # Mostrar el video con detecciones
        cv2.imshow("Juego de reconocimiento de objetos", result_frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Gracias por jugar. ¡Hasta luego!")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Iniciar el juego
if __name__ == "__main__":
    play_game()
