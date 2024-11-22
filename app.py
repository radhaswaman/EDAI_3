import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import h5py

# Function to download large files from Google Drive
def download_large_file_from_google_drive(file_id, destination):
    try:
        # Base URL for Google Drive downloads
        URL = "https://docs.google.com/uc?export=download"

        # Start session
        session = requests.Session()

        # Initial request to get the confirmation token
        response = session.get(URL, params={'id': file_id}, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Check for the presence of a warning cookie indicating large file download
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        # If a token is found, use it to get the full download
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file to the destination
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        st.success("Model downloaded successfully.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error during file download: {e}")
        raise  # Re-raise the exception to stop further processing
    except Exception as e:
        st.error(f"An unexpected error occurred while downloading the file: {e}")
        raise  # Re-raise the exception

# Function to ensure the model is downloaded
def download_model():
    model_file_id = 'https://drive.google.com/file/d/1DVMIbOppN2XlG38yV5l7AB_G0HhBHUyx/view?usp=sharing'  # Replace with your file ID
    output_path = 'breast_cancer_cnn_model.h5'
    if not os.path.exists(output_path):  # Check if the model is already downloaded
        with st.spinner("Downloading model..."):
            download_large_file_from_google_drive(model_file_id, output_path)
    return output_path

# Function to load the model with error handling
def load_model_safe(model_path):
    try:
        # Check if h5py is available for handling .h5 files
        if not h5py.is_hdf5(model_path):
            raise ValueError(f"Model file {model_path} is not a valid HDF5 file.")

        model = tf.keras.models.load_model(model_path)  # Try loading the model
        st.success("Model loaded successfully.")
        return model
    except ValueError as ve:
        st.error(f"Model loading error: {ve}")
    except OSError as oe:
        st.error(f"OS error during model loading: {oe}")
    except Exception as e:
        st.error(f"Unexpected error while loading model: {e}")
    return None

# Load the model
MODEL_PATH = download_model()
model = load_model_safe(MODEL_PATH)

# Preprocessing function for uploaded images
def load_and_preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(img_array):
    try:
        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
            return None

        prediction = model.predict(img_array)  # Model must be defined here
        return np.round(prediction[0][0])
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit App
st.title("Breast Cancer Image Classification")
st.write("Upload an image to classify it as Malignant or Benign.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    st.write("Classifying...")
    img_array = load_and_preprocess_image(img)
    prediction = predict_image(img_array)

    # Display result
    if prediction == 1:
        st.write("Prediction: Malignant")
    else:
        st.write("Prediction: Benign")
