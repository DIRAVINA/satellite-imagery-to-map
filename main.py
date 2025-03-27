from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import cv2
import os

app = FastAPI()

# Mount static directory for serving HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom Conv2DTranspose to ignore 'groups' parameter
class CustomConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Load the model with custom objects
print("Starting to load model...")
model = tf.keras.models.load_model(
    'model/GAN_Sat_image_2_map.h5',
    custom_objects={'Conv2DTranspose': CustomConv2DTranspose}
)
print("Model loaded successfully!")

# Preprocess image for the model
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Adjust to your model's input size
    img = img / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)

# Postprocess model output
def postprocess_image(output):
    output = (output[0] * 255).astype(np.uint8)  # Denormalize
    return output

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    input_path = "static/input.png"
    output_path = "static/output.png"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process with model
    input_img = preprocess_image(input_path)
    output_img = model.predict(input_img)
    result_img = postprocess_image(output_img)

    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    return {"image_url": "/static/output.png"}