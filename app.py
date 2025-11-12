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
CORS(app)  # Permite peticiones desde otros orígenes (frontend)

# Mapea categorías numéricas a nombres de carpeta
category_map = {
    0: "alegria",
    1: "tristeza",
    2: "enojo"
}

UPLOAD_FOLDER = "uploads"

# Asegura que las carpetas existan
for folder in category_map.values():
    os.makedirs(os.path.join(UPLOAD_FOLDER, folder), exist_ok=True)

@app.route('/save-drawing', methods=['POST'])
def save_drawing():
    data = request.get_json()
    image_data = data.get('image')
    category = data.get('category')

    if not image_data or category not in category_map:
        return jsonify({"error": "Datos inválidos"}), 400

    prefix = "data:image/png;base64,"
    if not image_data.startswith(prefix):
        return jsonify({"error": "Formato de imagen inválido"}), 400

    image_base64 = image_data[len(prefix):]

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        return jsonify({"error": "Error al decodificar la imagen"}), 400

    # ✅ Usa BytesIO del módulo `io`, no de skimage.io
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        resized_image = image.resize((256, 256))  # Puedes cambiar el tamaño aquí

        folder = category_map[category]
        filename = f"drawing_{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, folder, filename)

        resized_image.save(filepath, format="PNG")
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500

    return jsonify({"message": "Imagen guardada", "filename": filename})

@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    categories = ["alegria", "tristeza", "enojo"]
    labels = []

    for category in categories:
        filelist = glob.glob(f'uploads/{category}/*.png')
        if not filelist:
            continue

        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]  # Extraer canal alfa (transparencia)
        labels_read = np.array([category] * images_read.shape[0])

        images.append(images_read)
        labels.append(labels_read)

    if images:
        images = np.vstack(images)
        labels = np.concatenate(labels)
        np.save('X.npy', images)
        np.save('y.npy', labels)
        return jsonify({
            "message": "¡Dataset preparado con éxito!",
            "num_images": images.shape[0],
            "categories": categories
        })
    else:
        return jsonify({
            "error": "No se encontraron imágenes en las carpetas esperadas.",
            "num_images": 0,
            "categories": categories
        }), 404
@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('./X.npy')

@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file('./y.npy')

@app.route('/total-images', methods=['GET'])
def total_images():
    total = 0
    category_counts = {}

    for category in category_map.values():
        folder_path = os.path.join(UPLOAD_FOLDER, category)
        count = len(glob.glob(os.path.join(folder_path, "*.png")))
        category_counts[category] = count
        total += count

    return jsonify({
        "total_images": total,
        "images_per_category": category_counts
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
