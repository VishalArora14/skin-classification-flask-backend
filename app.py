from flask import Flask, request, jsonify #, render_template
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import decode_predictions
 
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19

from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

# Load your models
# model_path = 'short_psoriasis.h5'

# define class names
class_names = ['guttate_psoriasis', 'Nail_psoriasis', 'plaque_psoriasis']

# Get the directory where your app is running on Render.com
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'short_psoriasis.h5')
model = load_model(model_path)

vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
for layer in vgg_model.layers:
    layer.trainable = False

def preprocess_image(img):
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_skin_disease(img):
    img = preprocess_image(img)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    pred = model.predict(img)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name


@app.route('/', methods=['GET'])
def hello_word():
    return "Group 39 Minor"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    imagefile = request.files['image']
    img = cv2.imdecode(np.frombuffer(imagefile.read(), np.uint8), cv2.IMREAD_COLOR)

    predicted_class_name = predict_skin_disease(img)

    return jsonify({'prediction': predicted_class_name})


if __name__ == '__main__':
    app.run(port=8000, debug=True)