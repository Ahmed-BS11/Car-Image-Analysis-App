import streamlit as st
import numpy as np
import requests
from PIL import Image

# Page title and introduction
st.title("AI-Generated Image Detector")
st.write("Upload an image, and we'll determine if it's AI-generated or not.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the InceptionV3 model
    model = InceptionV3(weights='imagenet')

    # Preprocess the image
    img = image.resize((299, 299))  # Resize image to the required input size of InceptionV3
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make predictions
    prediction = model.predict(img)
    decoded_predictions = decode_predictions(prediction, top=1)[0][0]

    # Display the result
    if decoded_predictions[1].lower() == 'gazelle':
        result = "AI-Generated Image"
    else:
        result = "Not AI-Generated Image"

    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {decoded_predictions[2] * 100:.2f}%")