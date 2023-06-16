# import library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from io import BytesIO
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, json
import cv2
import base64
import firebase_admin
from google.cloud import firestore, storage
from firebase_admin import credentials, firestore, initialize_app


# Initializing the Google Cloud Storage Client
storage_client = storage.Client()

def get_credentials():
    bucket_name = 'firebase_credentials'
    blob_name = 'credentials.json'

    # Retrieve a blob object from Cloud Storage
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Read the file content as a string
    credentials_data = blob.download_as_text()

    return credentials_data

cred = credentials.Certificate(json.loads(get_credentials()))
firebase_admin.initialize_app(cred)
db = firestore.client()

myModel = keras.models.load_model('model.h5')

# label from model jagung.h5
label = ["Bercak", "Hawar", "Karat", "Sehat"]

app = Flask(__name__)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def upload_image_to_storage(file):
    # Initializing the Firebase storage client.
    bucket_name = "capstone-project-387201.appspot.com"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Generate a unique name for the image file.
    filename = file.filename
    
    # Upload the image file to Firebase Storage.
    blob = bucket.blob(filename)
    blob.upload_from_string(file.read(), content_type=file.content_type)
    
    # Get the access URL for the image file.
    image_url = blob.public_url
    
    return image_url

# Function to save the prediction results to Cloud Firestore
def prediction_result(data, file):
    # Mendapatkan referensi koleksi "history"
    collection_ref = db.collection('history')

    # Create a new document with automatically generated ID
    doc_ref = collection_ref.document()

    # Upload image file to Firebase Storage
    image_url = upload_image_to_storage(file)
    
    # Save prediction data to the document
    doc_ref.set({
        'class': data['class'],
        'confidence': data['confidence'],
        'image_url': image_url,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    print("The prediction results have been successfully stored in Cloud Firestore.")

@app.route("/", methods=["GET"])
def index():
    html_content = """
    <html>
    <head>
        <title>CureCorn App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #fff;
            margin: 0;
            padding: 0;
        }

        .container {
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }

        h1 {
            text-align: center;
            color: #333;
            font-size: 32px;
            margin-bottom: 20px;
        }
    </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to CureCorn Web Server App!</h1>
        </div>
    </body>
    </html>
    """
    return html_content

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get('file')
    image = read_file_as_image(file.read())
    img = cv2.resize(image,(150,150))
    img_batch = np.expand_dims(img, 0)
    
    predictions = myModel.predict(img_batch)

    predicted_class = label[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Organizing the prediction results data to be stored in Cloud Firestore.
    data = {
        'class': predicted_class,
        'confidence': float(confidence),
        'image': file.filename,
    }

    # Saving the prediction results to Cloud Firestore.
    prediction_result(data, file)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'image': file.filename,
    }

if __name__ == "__main__":
    app.run(debug=True)
