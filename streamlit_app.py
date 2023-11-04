import streamlit as st
import numpy as np
import requests
from PIL import Image

# Page title and introduction
st.title("AI-Generated Image Detector")
st.write("Upload an image, and we'll determine if it's AI-generated or not.")

# Upload image
uploaded_image = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the InceptionV3 model
    model = InceptionV3(weights='imagenet')

    # Preprocess the image
    img = image.resize((32, 32))  # Resize the image to your desired dimensions
    img = image.img_to_array(img)
    img = img / 255.0  # Apply the same rescaling as in your data generators
    img = np.expand_dims(img, axis=0)

    # Make predictions
    prediction = model.predict(img)

    # Display the result
    if prediction < 0.5:
        result = "AI-Generated Image"
    else:
        result = "Not AI-Generated Image"

    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {100 - prediction[0][0] * 100:.2f}%")
