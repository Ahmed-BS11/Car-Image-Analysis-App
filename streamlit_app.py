import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Set page title and favicon
st.set_page_config(
    page_title="AI Image Detector ",
    page_icon=":camera:",  # You can choose an appropriate emoji as the icon
    layout="wide"

)
# Custom CSS for styling


st.title("AI-Generated Image Detector")
st.markdown(
    "<p style='text-align: center;'>Upload an image, and we'll determine if it's AI-generated or not.</p>",
    unsafe_allow_html=True,
)


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
@st.cache_resource
def load_aiornot_model():
    file_path = os.path.abspath("gnet.h5")
    model = load_model(file_path,compile=False)
    optimizer = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def load_severity_model():
    file_path = os.path.abspath("model_eff.h5")
    model = load_model(file_path,compile=False)
    optimizer = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

page=st.sidebar.selectbox('Select Algorithm',['AIorNot','Damage Severity','Damaged Parts','Segmentation'])
st.sidebar.markdown("""---""")
st.sidebar.write('Created by Faidi Hamza, Cherif Jawhar & Ben Salem Ahmed')

if page == 'AIorNot':
    upload_columns=st.columns([2,1])
    file_upload=upload_columns[0].expander(label='Upload Your Image')
    uploaded_image = file_upload.file_uploader("Choose an image...", type=["jpg", "png", "jpeg",'webp'], key="file_uploader")
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        upload_columns[1].image(image, caption="Uploaded Image", use_column_width=True)
        # Preprocess the image
        # Resize the image to your desired dimensions
        img = image.resize((224, 224))
        img = np.array(img)
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)
        model=load_aiornot_model()
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


if page == 'Damage Severity':
    upload_columns=st.columns([2,1])
    file_upload=upload_columns[0].expander(label='Upload Your Image')
    uploaded_image = file_upload.file_uploader("Choose an image...", type=["jpg", "png", "jpeg",'webp'], key="file_uploader")
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        upload_columns[1].image(image, caption="Uploaded Image", use_column_width=True, width=600)
        # Preprocess the image
        # Resize the image to your desired dimensions
        img = image.resize((224, 224))
        img = np.array(img)
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)
        model=load_severity_model()
        prediction = model.predict(img)
        #st.write(f"Prediction: {prediction}")
        # Display the result
        damage_classes = ["Minor Damage", "Moderate Damage", "Severe Damage"]
        predicted_class = damage_classes[np.argmax(prediction)]
        st.markdown(predicted_class)



