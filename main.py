# import library
import os
import uuid 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from io import BytesIO
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import cv2
import base64
import firebase_admin
from google.cloud import firestore
from firebase_admin import credentials, firestore, initialize_app
from google.cloud import storage


cred = credentials.Certificate("credentials.json")
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
    # Inisialisasi klien penyimpanan Firebase
    bucket_name = "capstone-project-387201.appspot.com"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Buat nama unik untuk file gambar
    filename = file.filename
    
    # Upload file gambar ke Firebase Storage
    blob = bucket.blob(filename)
    blob.upload_from_string(file.read(), content_type=file.content_type)
    
    # Dapatkan URL akses ke file gambar
    image_url = blob.public_url
    
    return image_url

# Fungsi untuk menyimpan hasil prediksi ke Cloud Firestore
def simpan_hasil_prediksi(data, file):
    # Mendapatkan referensi koleksi "history"
    collection_ref = db.collection('history')

    # Membuat dokumen baru dengan ID yang dihasilkan secara otomatis
    doc_ref = collection_ref.document()

    # Mengunggah file gambar ke Firebase Storage
    image_url = upload_image_to_storage(file)
    
    # Menyimpan data hasil prediksi ke dalam dokumen
    doc_ref.set({
        'class': data['class'],
        'confidence': data['confidence'],
        'image_url': image_url,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    print("Hasil prediksi berhasil disimpan ke Cloud Firestore.")

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

        .nav {
            margin-bottom: 40px;
            text-align: center;
        }

        .nav a {
            margin-right: 20px;
            text-decoration: none;
            color: #333;
            font-size: 20px;
        }

        .form-container {
            text-align: center;
        }

        .form-container input[type="file"] {
            margin-bottom: 20px;
            display: block;
            width: 100%;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 18px;
        }

        .form-container button {
            padding: 15px 30px;
            background-color: #A0D8B3;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }

        .form-container button:hover {
            background-color: #47A992;
        }
    </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to CureCorn Web Server App!</h1>
            <div class="nav">
                <a href="/docs">API Documentation</a>
            </div>
            <div class="form-container">
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".jpg,.jpeg,.png">
                    <br>
                    <button type="submit">Upload and Predict</button>
                </form>
            </div>
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

    # Menyusun data hasil prediksi untuk disimpan ke Cloud Firestore
    data = {
        'class': predicted_class,
        'confidence': float(confidence),
        'image': file.filename,
    }

    # Menyimpan hasil prediksi ke Cloud Firestore
    simpan_hasil_prediksi(data, file)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'image': file.filename,
    }

if __name__ == "__main__":
    app.run(debug=True)
