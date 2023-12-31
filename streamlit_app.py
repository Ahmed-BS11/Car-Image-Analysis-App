import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import subprocess
import uuid
import glob
from roboflow import Roboflow
import io


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
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def load_severity_model():
    file_path = os.path.abspath("model_eff.h5")
    model = load_model(file_path,compile=False)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def load_automl():
    rf = Roboflow(api_key="EJdF3gB2PwrQDNlVhauC")
    project = rf.workspace().project("car-damage-coco-v9i")
    model = project.version(1).model
    return model

page=st.sidebar.selectbox('Select Algorithm',['AIorNot','Damage Severity','Damaged Parts','Segmentation'])
st.sidebar.markdown("""---""")
st.sidebar.write('Created by Faidi Hamza, Cherif Jawhar & Ben Salem Ahmed')

if page == 'AIorNot':
    upload_columns = st.columns([2, 1])
    file_upload = upload_columns[0].expander(label='Upload Your Image')
    uploaded_image = file_upload.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", 'webp'], key="file_uploader")

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        upload_columns[1].image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add a button for prediction
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Predict AI or Not"):
            # Preprocess the image
            img = image.resize((224, 224))
            img = np.array(img)
            img = img / 255.0  
            img = np.expand_dims(img, axis=0)
            
            # Load the model
            model = load_aiornot_model()
            prediction = model.predict(img)
            
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
    upload_columns = st.columns([2, 1])
    file_upload = upload_columns[0].expander(label='Upload Your Image')
    uploaded_image = file_upload.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", 'webp'], key="file_uploader")

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        upload_columns[1].image(image, caption="Uploaded Image", use_column_width=True, width=600)
        
        # Add a button for prediction
        col1, col2, col3 = st.columns([1,1,1]) # this used to center the button
        if col2.button("Predict Damage Severity"):
            # Preprocess the image
            img = image.resize((224, 224))
            img = np.array(img)
            img = img / 255.0  
            img = np.expand_dims(img, axis=0)
            
            # Load the model
            model = load_severity_model()
            prediction = model.predict(img)
            
            # Display the result
            damage_classes = ["Minor Damage", "Moderate Damage", "Severe Damage"]
            predicted_class = damage_classes[np.argmax(prediction)]
            confidence = prediction[0][np.argmax(prediction)]
            st.markdown(f"<p style='font-size:60px;'>Prediction: {predicted_class}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:40px;'>Confidence: {confidence * 100:.2f}%</p>", unsafe_allow_html=True)

if page == 'Damaged Parts':
    upload_columns = st.columns([2, 1])
    file_upload = upload_columns[0].expander(label='Upload Your Image')
    uploaded_image = file_upload.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", 'webp'], key="file_uploader")

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        upload_columns[1].image(image, caption="Uploaded Image", use_column_width=True, width=600)
        
        # Add a button for prediction
        col1, col2, col3 = st.columns([1,1,1]) # this used to center the button
        if col2.button("Predict Damaged Parts with YOLO"):
            # Save the image to a temporary directory with a unique filename
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"uploaded_image_{uuid.uuid4()}.jpg")
            image.save(temp_image_path, format='JPEG')

            # Run the YOLO command using subprocess with the image path
            command = f"yolo task=detect mode=predict model=best.pt conf=0.25 source={temp_image_path}"
            yolo_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Display the YOLO results
            if yolo_process.returncode == 0:
                # Get the latest subdirectory under "runs/detect"
                latest_subdir = max(glob.glob(os.path.join("runs", "detect", "predict*")), key=os.path.getctime)

                result_image_path = os.path.join(latest_subdir, f"{os.path.basename(temp_image_path).split('.')[0]}.jpg")
                result_image = Image.open(result_image_path)
                col2.image(result_image, caption="YOLO Result", use_column_width=True)

                # Remove the temporary directory after displaying the result
                #os.rmdir(temp_dir)
            else:
                st.error(f"YOLO process failed with error:\n{yolo_process.stderr.decode('utf-8')}")
        if col2.button("Segment Damaged Parts with Roboflow model"):
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"uploaded_image_{uuid.uuid4()}.jpg")
            image.save(temp_image_path, format='JPEG')

            model = load_automl()

            # Make a prediction request to the AutoML model
            model.predict(temp_image_path).save("prediction.jpg")

            # Display the prediction image in Streamlit
            col2.image("prediction.jpg", caption="Prediction Image", use_column_width=True)
