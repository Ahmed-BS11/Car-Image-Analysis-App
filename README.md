<h1 align="center">
  <br>
  Car Image Analysis App
</h1>

<div align="center">
  <h4>
    <a href="#overview">Overview</a> |
    <a href="#getting-started">Getting Started</a> |
    <a href="#usage">Usage</a> |
    <a href="#models-and-apis">Models and APIs</a> |
    <a href="#access-the-live-streamlit-app">Live Streamlit App</a> |
    <a href="#additional-information">Additional Information</a>
  </h4>
</div>

<br>



# Car Image Analysis App

Welcome to the Car Image Analysis Streamlit App! This interactive application allows you to analyze car images using various algorithms, including AI image detection, damage severity assessment, damaged parts detection, and repair cost estimation.

# Overview

The application consists of several functionalities:

### 1. AI Image Detection

Determine if the uploaded image is AI-generated or not.

### 2. Damage Severity Assessment

Predict the severity of the damage in the car image, such as Minor, Moderate, or Severe damage.

### 3. Damaged Parts Detection

Detect and highlight damaged parts in the car image using two different algorithms:
   - YOLO (You Only Look Once) for object detection
   - Roboflow's AutoML model for segmentation

### 4. Repair Cost Estimation

Estimate the repair cost based on the brand of the car, detected damage severity, and damaged parts. The application uses pretrained models for car brand recognition and damage detection from Hugging Face's model hub.

# Getting Started

Follow these steps to set up and run the Car Image Analysis Streamlit App:

1. Clone the repository:

   ```bash
   git clone https://github.com/Ahmed-BS11/Cycling-Stations-Monitoring.git
2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
3. Run the Streamlit app :
    ```bash
    streamlit run streamlit_app.py
4. Open your browser and navigate to the provided local URL to access the app.

# Usage

1. Choose the desired algorithm from the sidebar (AI Image Detection, Damage Severity, Damaged Parts, Repair Cost).
2. Upload a car image by clicking the "Choose an image..." button.
3. Analyze the image using the selected algorithm.

# Models and APIs

The application uses the following models and APIs:

- **AI Image Detection:**
  - Custom model trained to detect whether an image is AI-generated or not.
    - Trained models:
      - Convolutional Neural Network (CNN) model from scratch
      - Inception V3 model using the [AIData Kaggle dataset](https://www.kaggle.com/datasets/derrickmwiti/aidata)
      - Vision Transformer (VIT) model using the CIFake dataset
    - Best performing model: Inception V3

- **Damage Severity Assessment:**
  - Custom model trained to predict the severity of car damage using the [Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset).
    - Trained models:
      - ResNet
      - 4D model
      - EfficientNet 
    - Best performing model: EfficientNet

- **Damaged Parts Detection:**
  - All the algorithms are trained using the [Car Damage COCO Dataset](https://universe.roboflow.com/dan-vmm5z/car-damage-coco-dataset).
  - YOLOv8 for object detection.
  - Mask RCNN.
  - Another model that we have trained with Roboflow.
    - The YOLOv8 and the one trained with Roboflow are implemented in our interface
- **Price Prediction :**
For predicting repair costs, the application utilizes the following components:
  - We have employed the [Car Brand pretrained model](https://huggingface.co/dima806/car_brand_image_detection/tree/main/) from Hugging Face to accurately identify the brand of the car.
  - We have also incorporated the [Car damaged parts model](https://huggingface.co/dima806/car_brand_image_detection/tree/main/) from Hugging Face for detecting damaged parts in the car.
  - We have finally used our chosen damage severity model to enhance the precision of repair cost predictions.
  - Two dictionaries have been employed for mapping:
    - `car_types`: Maps car brands to their corresponding types, such as Luxury or Standard.
    - `repair_cost_by_type`: Maps car types, damage types, and severities to estimated repair costs.
  
# Access the Live Streamlit App

🚗 Explore the Car Image Analysis App hosted on Streamlit **[Car Image Analysis App](https://aiornot.streamlit.app/) 🚀**. Interact with the various algorithms and models to analyze car images effortlessly!


# Additional Information
- For more details on each algorithm, model training, and external dependencies, refer to the specific sections in the source code.
