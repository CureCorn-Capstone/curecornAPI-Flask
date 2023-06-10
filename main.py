# import library
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from io import BytesIO
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request
import cv2

myModel = keras.models.load_model('model.h5')

# label from model jagung.h5
label = ["Bercak", "Hawar", "Karat", "Sehat"]

app = Flask(__name__)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route("/pred", methods=["GET", "POST"])
def predict():
    file = request.files.get('file')
    image = read_file_as_image(file.read())
    img = cv2.resize(image,(150,150))
    img_batch = np.expand_dims(img, 0)
    
    predictions = myModel.predict(img_batch)

    predicted_class = label[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence),
    }

if __name__ == "__main__":
    app.run(debug=True)

