import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os


# Function to download the model from Google Drive
def download_model():
    model_file_id = '1DVMIbOppN2XlG38yV5l7AB_G0HhBHUyx'  # Replace with your file ID
    model_url = f'https://drive.google.com/uc?export=download&id={model_file_id}'
    output_path = 'best_model.keras'
    if not os.path.exists(output_path):  # Check if the model is already downloaded
        with st.spinner("Downloading model..."):
            gdown.download(model_url, output_path, quiet=False)
    return output_path

# Example usage
MODEL_PATH = download_model()
print(f"Model downloaded to: {MODEL_PATH}")

# Preprocessing function for uploaded images
def load_and_preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(img_array):
    prediction = model.predict(img_array)
    return np.round(prediction[0][0])

# Streamlit App
st.title("Breast Cancer Image Classification")
st.write("Upload an image to classify it as Malignant or Benign.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Classifying...")
    img_array = load_and_preprocess_image(img)
    prediction = predict_image(img_array)

    # Display result
    if prediction == 1:
        st.write("Prediction: Malignant")
    else:
        st.write("Prediction: Benign")
