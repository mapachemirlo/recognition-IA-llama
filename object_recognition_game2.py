# ---------------------------  v2
from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from groq_service import get_description_from_groq  # Importa la función

# Inicializar YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inicializar Flask
app = Flask(__name__)

# Obtener la cámara
cap = cv2.VideoCapture(0)

# Variable global para almacenar los objetos ya identificados y el historial de descripciones
identified_objects = set()
history = []

def generate_frames():
    global identified_objects, history  # Usamos las variables globales

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Realizar detección con YOLOv5
        results = model(frame)
        detections = results.pandas().xyxy[0]  # DataFrame con detecciones

        # Procesar cada objeto detectado
        for _, row in detections.iterrows():
            object_name = row['name']
            confidence = row['confidence']
            
            # Si el objeto ya fue identificado, no lo volvemos a procesar
            if object_name not in identified_objects and confidence > 0.5:  # Solo describir objetos con alta confianza
                identified_objects.add(object_name)  # Marcar el objeto como identificado
                description = get_description_from_groq(object_name)  # Obtener la descripción del objeto
                history.append({"object": object_name, "description": description})  # Agregar a la lista de historial

            # Dibujar las cajas de detección
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{object_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Codificar el frame para transmisión
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/describe_object')
def describe_object():
    """
    Endpoint para obtener la descripción del último objeto detectado.
    """
    global history
    return jsonify({"history": history})

if __name__ == '__main__':
    ##app.run(debug=True)  solo para modo debugger
    import os
    PORT = os.getenv('PORT', 5000)  # Por defecto, usará el puerto 5001
    app.run(host='0.0.0.0', port=int(PORT))



# ------------------------- V1 --------------------
# from flask import Flask, render_template, Response, jsonify, request
# import cv2
# import torch
# from groq_service import get_description_from_groq  # Importa la función

# # Inicializar YOLOv5
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Inicializar Flask
# app = Flask(__name__)

# # Obtener la cámara
# cap = cv2.VideoCapture(0)

# # Variable global para la descripción del último objeto detectado
# last_object_description = ""

# def generate_frames():
#     global last_object_description  # Usamos la variable global para la descripción

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Realizar detección con YOLOv5
#         results = model(frame)
#         detections = results.pandas().xyxy[0]  # DataFrame con detecciones

#         # Procesar cada objeto detectado
#         for _, row in detections.iterrows():
#             object_name = row['name']
#             confidence = row['confidence']
#             if confidence > 0.5:  # Solo describir objetos con alta confianza
#                 last_object_description = get_description_from_groq(object_name)  # Actualizamos la descripción

#             # Dibujar las cajas de detección
#             x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             label = f"{object_name} {confidence:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Codificar el frame para transmisión
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/describe_object')
# def describe_object():
#     """
#     Endpoint para obtener la descripción del último objeto detectado.
#     """
#     global last_object_description
#     return jsonify({"description": last_object_description})

# if __name__ == '__main__':
#     app.run(debug=True)


