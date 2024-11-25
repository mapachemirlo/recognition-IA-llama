from flask import Flask, render_template, Response, jsonify, request
import torch
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import base64
from groq_service import get_description_from_groq

# Inicializar YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Inicializar Flask
app = Flask(__name__)

# Variable global para almacenar los objetos ya identificados y el historial de descripciones
identified_objects = set()
history = []

def process_frame(frame):
    """
    Procesa el frame recibido (en formato PIL Image) y realiza la detección con YOLOv5.
    """
    global identified_objects, history

    # Convertir la imagen de PIL a numpy para usarla con YOLO
    img_np = np.array(frame)

    # Realizar detección con YOLOv5
    results = model(img_np)
    detections = results.pandas().xyxy[0]  # DataFrame con detecciones

    # Dibujar las cajas de detección sobre la imagen
    draw = ImageDraw.Draw(frame)

    # Procesar cada objeto detectado
    for _, row in detections.iterrows():
        object_name = row['name']
        confidence = row['confidence']
        
        # Si el objeto ya fue identificado, no lo volvemos a procesar
        if object_name not in identified_objects and confidence > 0.5:  # Solo describir objetos con alta confianza
            identified_objects.add(object_name)  # Marcar el objeto como identificado
            description = get_description_from_groq(object_name)  # Obtener la descripción del objeto
            history.append({"object": object_name, "description": description})  # Agregar a la lista de historial

        # Dibujar las cajas de detección sobre el frame
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{object_name} {confidence:.2f}"
        # Dibujar las cajas de detección sobre el frame
        draw.rectangle([x1, y1, x2, y2], outline="lightgreen", width=2)  # Cambiado a un color más claro
        draw.text((x1, y1 - 10), label, fill="lightgreen")  # Cambiado a un color más claro


    return frame  # Devuelvo el frame para la transmisión

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/video_feed', methods=['POST'])
def video_feed():
    """
    Endpoint para recibir las imágenes desde el navegador y procesarlas con YOLO.
    """
    image_data = request.form['image']
    image_data = image_data.split(',')[1]  # Extraer los datos base64 de la imagen
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))  # Convertir a imagen PIL

    processed_image = process_frame(image)

    # Convertir la imagen procesada de PIL a base64 para devolverla al navegador
    buffered = BytesIO()
    processed_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': f"data:image/jpeg;base64,{img_str}"})

@app.route('/describe_object')
def describe_object():
    """
    Endpoint para obtener la descripción del último objeto detectado.
    """
    global history
    return jsonify({"history": history})

if __name__ == '__main__':
    import os
    PORT = os.getenv('PORT', 10000)
    app.run(host='0.0.0.0', port=int(PORT))
