from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import random
import pickle
from werkzeug.utils import secure_filename
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='mri-images')
app.config['UPLOAD_FOLDER'] = os.path.join('mri-images', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CNN = None

def load_model():
    global CNN
    if CNN is not None:
        return
    script_dir = os.path.dirname(__file__)
    model_json_path = os.path.join(script_dir, 'models', 'CNN_structure.json')
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    try:
        CNN = tf.keras.models.model_from_json(model_json)
        weights_path = os.path.join(script_dir, 'models', 'CNN_weights.pkl')
        with open(weights_path, 'rb') as weights_file:
            weights = pickle.load(weights_file)
            CNN.set_weights(weights)
        CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_model_prediction(image_path):
    load_model()
    try:
        img = Image.open(image_path).resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.expand_dims(np.array(img), axis=0)  # No normalization: matches your original logic!
        prediction = CNN.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        predicted_class = class_labels[predicted_index]
        confidence_percent = round(confidence, 2)
        return predicted_class, confidence_percent
    except Exception as e:
        logger.error(f"Error in get_model_prediction: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction, confidence = get_model_prediction(filepath)
            web_accessible_path = url_for('static', filename=f'uploads/{filename}')
            return render_template('upload.html', filename=filename, prediction=prediction, confidence=confidence, img_path=web_accessible_path)
    return render_template('upload.html', filename=None)

@app.route('/get-random-image', methods=['GET'])
def get_random_image():
    try:
        class_dirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
        selected_class = random.choice(class_dirs)
        image_dir = os.path.join('mri-images', selected_class)
        image_list = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        image_name = random.choice(image_list)
        image_path = os.path.join(image_dir, image_name)
        predicted_label, confidence = get_model_prediction(image_path)
        web_accessible_image_path = url_for('static', filename=f'{selected_class}/{image_name}')
        return jsonify({
            'image_path': web_accessible_image_path,
            'actual_label': selected_class,
            'predicted_label': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Error in get-random-image route: {e}")
        return jsonify({'error': 'An error occurred', 'image_path':'', 'actual_label':'', 'predicted_label':'', 'confidence':''}), 500

if __name__ == '__main__':
    app.run(debug=False)