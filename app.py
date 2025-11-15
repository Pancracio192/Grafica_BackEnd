import os
import base64
import glob
from flask import Flask, request, jsonify, send_file
from PIL import Image
from flask_cors import CORS
import numpy as np
import uuid
from io import BytesIO 
from skimage import io as skio
import tensorflow as tf
from tensorflow import keras
import cv2
        

app = Flask(__name__)
CORS(app)

category_map = {
    0: "alegria",
    1: "enojo",
    2: "tristeza"
}

color_map = ["rojo", "azul", "verde"]

UPLOAD_FOLDER = "uploads"

for emotion in category_map.values():
    for color in color_map:
        os.makedirs(os.path.join(UPLOAD_FOLDER, emotion, color), exist_ok=True)

modelo_normal = None
modelo_pretrained = None

def cargar_modelos():
    """Carga los modelos entrenados al iniciar el servidor"""
    global modelo_normal, modelo_pretrained
    
    try:
        # Cargar modelo entrenado directamente
        modelo_normal = keras.models.load_model('modelo_normal.keras')
        print("Modelo normal cargado")
    except Exception as e:
        print(f"No se pudo cargar modelo normal: {e}")
    
    try:
        # Cargar modelo pre-entrenado con MNIST
        modelo_pretrained = keras.models.load_model('modelo_pretrained.keras')
        print("Modelo pre-entrenado cargado")
    except Exception as e:
        print(f"No se pudo cargar modelo pre-entrenado: {e}")

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

        image.save(filepath, format="PNG")
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

            images_in_folder = []
            for img_path in filelist:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                images_in_folder.append(img_array)
            
            if not images_in_folder:
                continue
            
            images_array = np.stack(images_in_folder)
            num_images = images_array.shape[0]
            
            emotion_labels = np.array([emotion] * num_images)
            color_labels = np.array([color] * num_images)

            images.append(images_array)
            labels_emotion.append(emotion_labels)
            labels_color.append(color_labels)

    if images:
        images = np.vstack(images)
        labels_emotion = np.concatenate(labels_emotion)
        labels_color = np.concatenate(labels_color)
        
        labels = np.column_stack((labels_emotion, labels_color))
        
        np.save('X.npy', images)
        np.save('y.npy', labels)
        
        return jsonify({
            "message": "¡Dataset preparado con éxito!",
            "num_images": images.shape[0],
            "image_shape": list(images.shape),
            "emotions": list(category_map.values()),
            "colors": color_map
        })
    else:
        return jsonify({
            "error": "No se encontraron imágenes",
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

@app.route('/predict', methods=['POST'])
def predict():
    """Predice la emoción de una imagen dibujada"""
    data = request.get_json()
    image_data = data.get('image')
    model_type = data.get('model_type', 'normal')
    
    if not image_data:
        return jsonify({"error": "No se proporcionó imagen"}), 400
    
    modelo = modelo_normal if model_type == 'normal' else modelo_pretrained
    
    if modelo is None:
        return jsonify({"error": f"Modelo {model_type} no disponible"}), 500
    
    try:
        # Decodificar imagen
        prefix = "data:image/png;base64,"
        if image_data.startswith(prefix):
            image_data = image_data[len(prefix):]
        
        image_bytes = base64.b64decode(image_data)
        img_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_rgb = np.array(img_pil)        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        from skimage.transform import resize
        img_resized = resize(img_gray, (28, 28), anti_aliasing=True, preserve_range=True)
        img_resized = img_resized.astype('float32') / 255.0
        img_final = img_resized.reshape(1, 28, 28, 1)
        
        print(f"🔍 Shape: {img_final.shape}, Min: {img_final.min():.3f}, Max: {img_final.max():.3f}")
        
        # Predecir
        predictions = modelo.predict(img_final, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        emotion_name = category_map[predicted_class]
        
        print(f"Predicción: {emotion_name} ({confidence:.2%})")
        
        return jsonify({
            "emotion": emotion_name,
            "confidence": confidence,
            "model_type": model_type,
            "all_predictions": {
                "alegria": float(predictions[0][0]),
                "tristeza": float(predictions[0][1]),
                "enojo": float(predictions[0][2])
            }
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    cargar_modelos()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

