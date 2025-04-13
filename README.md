
# Satellite to Map Translation using Pix2Pix GAN

This project uses **Pix2Pix GAN** (Generative Adversarial Network) to convert **satellite images** into **map-like representations**. The model consists of a **U-Net Generator** and **PatchGAN Discriminator** to learn the mapping between satellite imagery and map imagery.

## Table of Contents

1. [Project Overview](#project-overview)
2. [How to Run](#how-to-run)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Inference & Testing](#inference--testing)
6. [Web Deployment](#web-deployment)
7. [Challenges & Lessons Learned](#challenges--lessons-learned)
8. [Future Work](#future-work)
9. [References](#references)

## Project Overview

This project demonstrates the process of training a **Pix2Pix GAN** model to generate map-like images from satellite images. The architecture includes:
- **Generator**: A **U-Net** model with skip connections to preserve spatial structure.
- **Discriminator**: A **PatchGAN** classifier to distinguish between real and fake map images.
- **Loss Functions**: A combination of **Adversarial Loss** and **L1 Pixel-wise Loss** to guide training.
- **Web Interface**: A FastAPI-based web application to fetch satellite images, process them with the trained model, and display the results.

## How to Run

To run the code and start using the model, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/satellite-to-map-translation.git
cd satellite-to-map-translation
```

### 2. Install Dependencies
Make sure you have Python installed (recommended version: **Python 3.7+**). Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Here’s a breakdown of the essential dependencies:
- `tensorflow`: For model building and training.
- `fastapi`: For deploying the trained model as a web API.
- `requests`: For fetching satellite images via API.
- `opencv-python`: For image processing tasks like resizing and saving images.

### 3. Dataset Preparation
Make sure you have the **satellite-to-map dataset** ready. You can find the dataset structure inside the `maps/` directory. It should contain:
- `train/`: Paired satellite and map images for training.
- `val/`: Paired satellite and map images for validation.

If you're using a custom dataset, ensure it follows the same structure, with satellite images on the left and map images on the right.

To load the dataset and preprocess the images, you can call the `DatasetLoader` class, as shown in the code.

### 4. Training the Model
To train the model, run the following command:

```bash
python train.py
```

The training will:
- Load the dataset and preprocess the images.
- Train the **Generator** and **Discriminator** models for **100 epochs**.
- Periodically save the model weights in the `model/` directory.

### 5. Running Inference
After training the model, you can run inference to generate maps from satellite images using the `predict.py` script:

```bash
python predict.py
```

It will:
- Load the trained model.
- Use validation images to make predictions.
- Display the satellite image, ground truth map, and predicted map side by side.

## Web Deployment

You can also deploy the model using **FastAPI** for real-time predictions. Here's how:

### 1. Run the FastAPI Server
To start the web server for handling image uploads and map generation requests:

```bash
python main.py
```

The FastAPI server will run on **http://127.0.0.1:8000**.

### 2. Upload Image or Fetch Satellite Image
- Go to **http://127.0.0.1:8000/upload** to upload a satellite image and get the corresponding map.
- Go to **http://127.0.0.1:8000/fetch** to fetch a satellite image using **latitude** and **longitude**.

### 3. Web Interface Workflow
- **Fetch Image**: Users can input latitude, longitude, and radius to fetch satellite images from the Google Static Maps API (or OpenStreetMap).
- **Process Image**: After fetching or uploading the image, the FastAPI backend processes it with the Pix2Pix model to generate the map image.
- **Display Results**: The frontend displays both the original satellite image and the predicted map.

## Challenges & Lessons Learned
- **Data Quality**: Ensuring the paired images (satellite and map) were aligned correctly during the preprocessing stage was crucial for training stability.
- **Model Stability**: Balancing the loss between adversarial loss and L1 loss required careful tuning of the hyperparameters.
- **Training Time**: Training the model on larger datasets took considerable time, and optimizing the architecture further could speed up training.

## Future Work
- **Model Optimization**: Experimenting with architectures like **CycleGAN** to handle unpaired image translation.
- **Web Interface Improvements**: Adding more features like real-time image annotation and batch processing of multiple images.
- **Deployment on Cloud**: Deploying the model on platforms like **Heroku**, **AWS**, or **Google Cloud** for scalability and public access.

## References
1. **Isola et al. (2017)**. Image-to-Image Translation with Conditional Adversarial Networks. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
2. **Pix2Pix GitHub Repository**: [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)
3. **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
4. **FastAPI Documentation**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
