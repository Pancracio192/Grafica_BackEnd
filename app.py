import os
import base64
import glob
from flask import Flask, request, jsonify, send_file
from PIL import Image
from flask_cors import CORS
import numpy as np
import uuid
from io import BytesIO 
from skimage import io

app = Flask(__name__)
CORS(app)

category_map = {
    0: "alegria",
    1: "tristeza",
    2: "enojo"
}

color_map = ["rojo", "azul", "verde"]

UPLOAD_FOLDER = "uploads"

for emotion in category_map.values():
    for color in color_map:
        os.makedirs(os.path.join(UPLOAD_FOLDER, emotion, color), exist_ok=True)

@app.route('/save-drawing', methods=['POST'])
def save_drawing():
    data = request.get_json()
    image_data = data.get('image')
    category = data.get('category')
    color = data.get('color') 

    if not image_data or category not in category_map or color not in color_map:
        return jsonify({"error": "Datos inválidos"}), 400

    prefix = "data:image/png;base64,"
    if not image_data.startswith(prefix):
        return jsonify({"error": "Formato de imagen inválido"}), 400

    image_base64 = image_data[len(prefix):]

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        return jsonify({"error": "Error al decodificar la imagen"}), 400

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        emotion_folder = category_map[category]
        filename = f"drawing_{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, emotion_folder, color, filename)

        image.save(filepath, format="PNG")  # ✅ Guardar directo (ya es 32x32)
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500

    return jsonify({
        "message": "Imagen guardada", 
        "filename": filename,
        "emotion": emotion_folder,
        "color": color
    })

@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    labels_emotion = []
    labels_color = []

    for emotion in category_map.values():
        for color in color_map:
            folder_path = f'uploads/{emotion}/{color}'
            filelist = glob.glob(f'{folder_path}/*.png')
            
            if not filelist:
                continue

            images_read = io.concatenate_images(io.imread_collection(filelist))
            images_read = images_read[:, :, :, 3]  # Canal alfa
            
            num_images = images_read.shape[0]
            emotion_labels = np.array([emotion] * num_images)
            color_labels = np.array([color] * num_images)

            images.append(images_read)
            labels_emotion.append(emotion_labels)
            labels_color.append(color_labels)

    if images:
        images = np.vstack(images)
        labels_emotion = np.concatenate(labels_emotion)
        labels_color = np.concatenate(labels_color)
        
        # Combinar ambas etiquetas en un array 2D
        labels = np.column_stack((labels_emotion, labels_color))
        
        # Guardar X.npy y un solo y.npy con ambas columnas
        np.save('X.npy', images)
        np.save('y.npy', labels)  # Array 2D: [emoción, color]
        
        return jsonify({
            "message": "¡Dataset preparado con éxito!",
            "num_images": images.shape[0],
            "emotions": list(category_map.values()),
            "colors": color_map
        })
    else:
        return jsonify({
            "error": "No se encontraron imágenes en las carpetas esperadas.",
            "num_images": 0
        }), 404
    
@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('./X.npy')

@app.route('/y.npy', methods=['GET'])
def download_y_emotion():
    return send_file('./y.npy')

@app.route('/total-images', methods=['GET'])
def total_images():
    total = 0
    breakdown = {}

    for emotion in category_map.values():
        breakdown[emotion] = {}
        for color in color_map:
            folder_path = os.path.join(UPLOAD_FOLDER, emotion, color)
            count = len(glob.glob(os.path.join(folder_path, "*.png")))
            breakdown[emotion][color] = count
            total += count

    return jsonify({
        "total_images": total,
        "breakdown": breakdown
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)