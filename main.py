from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Disable GPU (since M1 uses Metal/CPU, not NVIDIA CUDA)
tf.config.set_visible_devices([], 'GPU')

app = FastAPI()

# Mount static directory for serving HTML and images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom Conv2DTranspose to ignore 'groups' parameter (optional)
class CustomConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Load the model
print("Starting to load model...")
try:
    model = tf.keras.models.load_model(
        'model/GAN_Generator.keras',
        custom_objects={'Conv2DTranspose': CustomConv2DTranspose}
    )
    print("Model loaded successfully!")
    sample_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    sample_output = model.predict(sample_input, verbose=0)
    print(f"Model output shape: {sample_output.shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

# Preprocess image (resize to 256x256 and normalize to [-1, 1])
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")
    original_height, original_width = img.shape[:2]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = (img / 127.5) - 1.0
    return np.expand_dims(img, axis=0).astype(np.float32), original_height, original_width

# Postprocess model output (resize back to original dimensions)
def postprocess_image(output, original_height, original_width):
    if output.shape[1:] != (256, 256, 3):
        raise ValueError(f"Unexpected output shape: {output.shape[1:]}")
    output = (output[0] * 0.5 + 0.5) * 255
    output = output.astype(np.uint8)
    output = cv2.resize(output, (original_width, original_height), interpolation=cv2.INTER_AREA)
    return output

# Fetch satellite image
def fetch_satellite_image(lat: float, lon: float, radius: float):
    deg_per_km = 1 / 111.32
    buffer = radius * deg_per_km
    min_lat, max_lat = lat - buffer, lat + buffer
    min_lon, max_lon = lon - buffer, lon + buffer

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=640x640&maptype=satellite&visible={min_lat},{min_lon}|{max_lat},{max_lon}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch satellite image")

    satellite_path = "static/satellite.png"
    with open(satellite_path, "wb") as f:
        f.write(response.content)
    return "/static/satellite.png"

@app.get("/", response_class=HTMLResponse)
async def serve_main():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/upload", response_class=HTMLResponse)
async def serve_upload():
    with open("static/upload.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/fetch", response_class=HTMLResponse)
async def serve_fetch():
    with open("static/fetch.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    input_path = "static/input.png"
    output_path = "static/output.png"

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        input_img, original_height, original_width = preprocess_image(input_path)
        output_img = model.predict(input_img, verbose=0)
        result_img = postprocess_image(output_img, original_height, original_width)
        print(f"Original input shape: {cv2.imread(input_path).shape}")
        print(f"Processed input shape: {input_img.shape[1:]}")
        print(f"Output shape: {output_img.shape[1:]}")
        print(f"Resized output shape: {result_img.shape}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save output image: {e}")

    return {"input_image_url": "/static/input.png", "output_image_url": "/static/output.png"}

@app.post("/fetch-satellite")
async def fetch_satellite(lat: float = Form(...), lon: float = Form(...), radius: float = Form(0.5)):
    try:
        satellite_url = fetch_satellite_image(lat, lon, radius)
        return {"satellite_image_url": satellite_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch satellite image: {e}")

@app.post("/process-satellite")
async def process_satellite():
    input_path = "static/satellite.png"
    output_path = "static/output.png"

    try:
        input_img, original_height, original_width = preprocess_image(input_path)
        output_img = model.predict(input_img, verbose=0)
        result_img = postprocess_image(output_img, original_height, original_width)
        print(f"Original input shape: {cv2.imread(input_path).shape}")
        print(f"Processed input shape: {input_img.shape[1:]}")
        print(f"Output shape: {output_img.shape[1:]}")
        print(f"Resized output shape: {result_img.shape}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save output image: {e}")

    return {"output_image_url": "/static/output.png"}

@app.post("/reset")
async def reset():
    input_path = "static/input.png"
    output_path = "static/output.png"
    satellite_path = "static/satellite.png"
    try:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(satellite_path):
            os.remove(satellite_path)
        print("Reset: Cleared input.png, output.png, and satellite.png")
        return {"message": "Reset successful", "redirect": "/"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")