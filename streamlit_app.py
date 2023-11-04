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

# Center-align the title and intro text
st.title("AI-Generated Image Detector")
st.markdown(
    "<p style='text-align: center;'>Upload an image, and we'll determine if it's AI-generated or not.</p>",
    unsafe_allow_html=True,
)

# Upload image with custom styling
uploaded_image = st.file_uploader("Choose an image...", type=[
                                  "jpg", "png", "jpeg"], key="file_uploader")

# Add some spacing for better visual separation
st.write("")

# Optionally, you can add more text or explanations to guide the user
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
    img = img / 255.0  # Apply the same rescaling as in your data generators
    img = np.expand_dims(img, axis=0)

    # Make predictions
    prediction = model.predict(img)

    # Display the result
    if prediction < 0.5:
        result = "AI-Generated Image"
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {100 - prediction[0][0] * 100:.2f}%")
    else:
        result = "Not AI-Generated Image"
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {prediction[0][0] * 100:.2f}%")

    
