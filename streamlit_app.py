import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import subprocess
import uuid
import glob
import requests

try:
    from roboflow import Roboflow
except ImportError as e:
    if "libGL.so.1" in str(e):
        print("Warning: Unable to import 'roboflow' due to missing libGL.so.1. Some functionality may be limited.")
    else:
        raise

# Set page title and favicon
st.set_page_config(
    page_title="Car Image Analysis App",
    page_icon="🚗",  # Customized icon representing cars
    layout="wide",
    initial_sidebar_state="expanded",  # Set to "collapsed" if you prefer a collapsed sidebar initially
)
# Custom CSS for styling

st.title("Car Image Analysis App")

# Common instructions for all pages
st.markdown(
    "<p style='text-align: center;'>Upload an image and utilize various algorithms to analyze car-related information.</p>",
    unsafe_allow_html=True,
)

# Instructions based on the selected algorithm
st.markdown(
    "#### Instructions",
    unsafe_allow_html=True,
)
st.markdown(
    "1. Click the 'Choose an image...' button to upload an image file (JPEG, PNG, or JPG).",
    unsafe_allow_html=True,
)
st.markdown(
    "2. Analyze the image using the selected algorithm.",
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

headers = {"Authorization": "Bearer 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"}
API_URL1 = "https://api-inference.huggingface.co/models/dima806/car_brand_image_detection"
API_URL2 = "https://api-inference.huggingface.co/models/beingamit99/car_damage_detection"

def query(filename,API_URL):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()
car_types = {
    'Aston Martin': 'Luxury',
    'Mercedes-Benz': 'Luxury',
    'Mini': 'Standard',
    'Tesla': 'Electric',
    'GMC': 'SUV',
    'Alfa Romeo': 'Sport',
    'Studebaker': 'Classic',
    'Suzuki': 'Standard',
    'Peugeot': 'Standard',
    'Genesis': 'Luxury',
    'BMW': 'Luxury',
    'Honda': 'Standard',
    'Chrysler': 'Standard',
    'Mazda': 'Standard',
    'Infiniti': 'Luxury',
    'Land Rover': 'SUV',
    'Dodge': 'Standard',
    'Fiat': 'Standard',
    'Maserati': 'Luxury',
    'Saab': 'Standard',
    'Nissan': 'Standard',
    'Hudson': 'Classic',
    'Lincoln': 'Luxury',
    'Volvo': 'Luxury',
    'Mitsubishi': 'Standard',
    'Oldsmobile': 'Classic',
    'Lexus': 'Luxury',
    'Buick': 'Luxury',
    'Jaguar': 'Luxury',
    'Toyota': 'Standard',
    'Volkswagen': 'Standard',
    'Renault': 'Standard',
    'Citroen': 'Standard',
    'Audi': 'Luxury',
    'Subaru': 'Standard',
    'Cadillac': 'Luxury',
    'Pontiac': 'Standard',
    'Porsche': 'Sport',
    'Daewoo': 'Standard',
    'Bugatti': 'Exotic',
    'Jeep': 'SUV',
    'Ram Trucks': 'Truck',
    'Chevrolet': 'Standard',
    'MG': 'Sport',
    'Hyundai': 'Standard',
    'Ferrari': 'Exotic',
    'Acura': 'Luxury',
    'Kia': 'Standard',
    'Bentley': 'Luxury',
    'Ford': 'Standard',
}
repair_cost_by_type = {
    'Luxury': {
        'Dent': {'Minor': 500, 'Moderate': 1000, 'Severe': 2000},
        'Scratch': {'Minor': 300, 'Moderate': 700, 'Severe': 1500},
        'Crack': {'Minor': 800, 'Moderate': 1200, 'Severe': 2500},
        'Glass Shatter': {'Minor': 1000, 'Moderate': 1800, 'Severe': 3000},
        'Lamp Broken': {'Minor': 600, 'Moderate': 1000, 'Severe': 2000},
    },
    'Standard': {
        'Dent': {'Minor': 400, 'Moderate': 850, 'Severe': 1600},
        'Scratch': {'Minor': 200, 'Moderate': 550, 'Severe': 1200},
        'Crack': {'Minor': 700, 'Moderate': 1050, 'Severe': 2200},
        'Glass Shatter': {'Minor': 800, 'Moderate': 1500, 'Severe': 2700},
        'Lamp Broken': {'Minor': 500, 'Moderate': 900, 'Severe': 1800},
    },
    'Sport': {
        'Dent': {'Minor': 600, 'Moderate': 1100, 'Severe': 2100},
        'Scratch': {'Minor': 350, 'Moderate': 800, 'Severe': 1600},
        'Crack': {'Minor': 900, 'Moderate': 1300, 'Severe': 2600},
        'Glass Shatter': {'Minor': 1100, 'Moderate': 2000, 'Severe': 3200},
        'Lamp Broken': {'Minor': 700, 'Moderate': 1200, 'Severe': 2300},
    },
    'Electric': {
        'Dent': {'Minor': 700, 'Moderate': 1200, 'Severe': 2300},
        'Scratch': {'Minor': 400, 'Moderate': 900, 'Severe': 1800},
        'Crack': {'Minor': 1000, 'Moderate': 1400, 'Severe': 2700},
        'Glass Shatter': {'Minor': 1200, 'Moderate': 2200, 'Severe': 3400},
        'Lamp Broken': {'Minor': 800, 'Moderate': 1300, 'Severe': 2400},
    },
    'SUV': {
        'Dent': {'Minor': 500, 'Moderate': 950, 'Severe': 1800},
        'Scratch': {'Minor': 250, 'Moderate': 600, 'Severe': 1300},
        'Crack': {'Minor': 800, 'Moderate': 1100, 'Severe': 2200},
        'Glass Shatter': {'Minor': 900, 'Moderate': 1600, 'Severe': 2800},
        'Lamp Broken': {'Minor': 550, 'Moderate': 1000, 'Severe': 2000},
    },
    'Classic': {
        'Dent': {'Minor': 300, 'Moderate': 700, 'Severe': 1500},
        'Scratch': {'Minor': 150, 'Moderate': 500, 'Severe': 1100},
        'Crack': {'Minor': 600, 'Moderate': 1000, 'Severe': 2000},
        'Glass Shatter': {'Minor': 700, 'Moderate': 1300, 'Severe': 2500},
        'Lamp Broken': {'Minor': 400, 'Moderate': 800, 'Severe': 1700},
    },
    # Add more types and their repair cost details...
}
st.sidebar.markdown("### Project description :")

st.sidebar.write(
    """
    This interactive application allows you to analyze car images using various algorithms.
    Choose from the options on the left to explore different functionalities.
    """
)
page=st.sidebar.selectbox('### Select Algorithm',['AIorNot','Damage Severity','Damaged Parts','Repair cost'])
st.sidebar.markdown("""---""")
st.sidebar.write('Created by Faidi Hamza, Cherif Jawhar & Ben Salem Ahmed')

st.sidebar.markdown("### Github Repo")
st.sidebar.markdown(
    f"<a href='https://github.com/Ahmed-BS11/AIorNOT' target='_blank'><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' alt='GitHub Repo' style='width: 50px; border-radius: 50%;'></a>",
    unsafe_allow_html=True,
)
if page == 'AIorNot':
    st.markdown(
        "3. Determine if the image is AI-generated or not.",
        unsafe_allow_html=True,
    )
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
            if prediction > 0.7:
                result = "AI-Generated Image"
                st.markdown(f"<p style='font-size:60px;'>Prediction: {result}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:40px;'>Confidence: {prediction[0][0] * 100:.2f}%</p>", unsafe_allow_html=True)
            else:
                result = "Not AI-Generated Image"
                st.markdown(f"<p style='font-size:60px;'>Prediction: {result}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:40px;'>Confidence: {100-prediction[0][0] * 100:.2f}%</p>", unsafe_allow_html=True)

if page == 'Damage Severity':
    st.markdown(
        "3. Predict the severity of the car damage.",
        unsafe_allow_html=True,
    )
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
    st.markdown(
        "3. Detect and highlight the damaged parts in the image.",
        unsafe_allow_html=True,
    )
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


if page == 'Repair cost':
    st.markdown(
        "3. Estimate the repair cost.",
        unsafe_allow_html=True,
    )
    upload_columns = st.columns([2, 1])
    file_upload = upload_columns[0].expander(label='Upload Your Image')
    uploaded_image = file_upload.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", 'webp'], key="file_uploader")

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        upload_columns[1].image(image, caption="Uploaded Image", use_column_width=True, width=600)
        
        # Add a button for prediction
        col1, col2, col3 = st.columns([1,1,1]) # this used to center the button
        if col2.button("Predict repair cost"):
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"uploaded_image_{uuid.uuid4()}.jpg")
            image.save(temp_image_path, format='JPEG')
            brand=query(temp_image_path,API_URL1)[0]['label']
            damage=query(temp_image_path,API_URL2)[0]['label']
            img = image.resize((224, 224))
            img = np.array(img)
            img = img / 255.0  
            img = np.expand_dims(img, axis=0)
            
            # Load the model
            model = load_severity_model()
            prediction = model.predict(img)
            
            # Display the result
            damage_classes = ["Minor", "Moderate", "Severe"]
            predicted_class = damage_classes[np.argmax(prediction)]
            car_info_html = (
                f"<p style='font-size:24px;'>Your car is a: "
                f"<span style='font-size:30px; color: #ff6666;'>{brand}</span>, which is a "
                f"<span style='font-size:30px; color: #33cc33;'>{car_types[brand]}</span> car. "
                f"Your car has a <span style='font-size:30px; color: #ffcc00;'>{predicted_class} {damage}</span> damage. "
                f"The cost of repair is <span style='font-size:36px; color: #ff3300;'>{repair_cost_by_type[car_types[brand]][damage][predicted_class]}$</span>.</p>"
            )

            st.markdown(car_info_html, unsafe_allow_html=True)

