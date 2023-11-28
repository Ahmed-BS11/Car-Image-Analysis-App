import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model

# Set page title and favicon
st.set_page_config(
    page_title="AI Image Detector ",
    page_icon=":camera:",  # You can choose an appropriate emoji as the icon
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;  /* Background color for the whole app */
    }
    .st-eb {
        background-color: #0077b6;  /* Background color for the file uploader */
        color: white;  /* Text color for the file uploader */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI-Generated Image Detector")
st.markdown(
    "<p style='text-align: center;'>Upload an image, and we'll determine if it's AI-generated or not.</p>",
    unsafe_allow_html=True,
)

uploaded_image = st.file_uploader("Choose an image...", type=[
                                  "jpg", "png", "jpeg",'webp'], key="file_uploader")


st.write("")

st.markdown(
    "#### Instructions",
    unsafe_allow_html=True,
)
st.markdown(
    "1. Click the 'Choose an image...' button to upload an image file (JPEG, PNG, or JPG).",
    unsafe_allow_html=True,
)
st.markdown(
    "2. We'll analyze the image and determine if it's AI-generated or not.",
    unsafe_allow_html=True,
)
from roboflow import Roboflow
import matplotlib.pyplot as plt
from PIL import Image

rf = Roboflow(api_key="EJdF3gB2PwrQDNlVhauC")
project = rf.workspace().project("car-damage-coco-dataset")
model = project.version(4).model
IM=model.predict("car4.jpg")
st.image(IM, caption="Uploaded Image", use_column_width=True)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    file_path = os.path.abspath("gnet.h5")
    model = load_model(file_path)

    # Preprocess the image
    # Resize the image to your desired dimensions
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    #st.write(f"Prediction: {prediction}")
    # Display the result
    if prediction > 0.5:
        result = "AI-Generated Image"
        st.markdown(f"<p style='font-size:60px;'>Prediction: {result}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:40px;'>Confidence: {prediction[0][0] * 100:.2f}%</p>", unsafe_allow_html=True)

    else:
        result = "Not AI-Generated Image"
        st.markdown(f"<p style='font-size:60px;'>Prediction: {result}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:40px;'>Confidence: {100-prediction[0][0] * 100:.2f}%</p>", unsafe_allow_html=True)

