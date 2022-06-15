import numpy as np
import csv
import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub


classifier_model = tf.keras.models.load_model('MobileNet')
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])
with open('70Food.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    FOOD = list(csv_reader)
    FoodMap = {}
    for i in FOOD:
        FoodMap[i[0]] = {
            "kcal": i[1],
            "fiber":i[2],
            "sodium":i[3],
            "saturated":i[4],
            "cholesterol":i[5],
            "omega":i[6]
        }

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.route("/image", methods=["POST"])
def process_image():
    file = request.files['image']
    img = Image.open(file.stream).resize(IMAGE_SHAPE)
    img = np.array(img)/255.0
    result = classifier.predict(img[np.newaxis, ...])
    predicted_class = tf.math.argmax(result[0], axis=-1)
    food_name = FOOD[predicted_class][0]
    output = {
        "name": food_name,
        "info": FoodMap[food_name],
    }
    return jsonify(output)  

@app.route("/name", methods=["POST"])
def process_name():
    name = request.data.decode("utf-8")
    if name in FoodMap:
        output = {
            name: FoodMap[name]
        }
    else:
        output = {
            name:  {
                "kcal":'-',
                "fiber":'-',
                "sodium":'-',
                "saturated":'-',
                "cholesterol":'-',
                "omega":'-'
            }
        }
    return jsonify(output )

app.run(host='0.0.0.0', port=80)
