<h2 align="center">Corn Disease Detection API</h1>

<div align="center">
    🌐 API Server: https://curecornapi-flask-afm43kqnba-uc.a.run.app/
</div>

> ⚠️ This API provides a service to detect diseases in corn plants based on the provided image. The application uses the Flask framework with Python programming language and is deployed on the Cloud Run platform for scalability and high availability.

## Endpoint
'/predict'

*Method:* POST

This endpoint is used to send an image of a corn plant and receive the disease prediction result.

## Request
- Content Type: 'application/json'
- Request Body: 'image': Corn plant image file to be processed (format: JPEG, PNG)

## Response
- Content Type: 'multipart/form-data'
- Request Body: 
```json

     {
        "class": "Healty",
        "confidence": 1.0,
        "image": "0a2dec45-729b-4825-b814-a73d14e8c7fe___R.S_HL 8211 copy.jpg"
      }
 ```
 
 ## Example Use:
 ```bash
 curl -X POST -F "image=@corn_image.jpg" https://curecornapi-flask-afm43kqnba-uc.a.run.app/predict
 ```
## Development
If you want to run the API locally for development purposes, follow these steps:

1. Ensure you have Python and pip installed on your system.
2. Clone this repository to your local system.
3. Open a terminal and navigate to the project directory.
4. Create a virtual environment and activate it
5. Install the required dependencies:
 ```bash
pip install -r requirements.txt
 ```
6. Run the API application locally:
 ```bash
python main.py
 ```
7. The API will run at http://localhost:5000.
