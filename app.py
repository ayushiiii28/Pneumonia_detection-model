import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Load the model using pickle
with open('training_history.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app layout
st.title("Image Classification with TensorFlow and Streamlit")
st.write("Upload an image to classify using your model.")

# Add some custom HTML content (optional)
st.markdown("""
    <div style="text-align:center;">
        <h2>Welcome to the Image Classification App</h2>
        <p>This app allows you to upload an image and classify it using a pre-trained TensorFlow model.</p>
    </div>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess and make predictions
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Display predictions
    st.write("Prediction:", predictions)

# Add footer (optional)
st.markdown("""
    <footer style="text-align:center; padding-top:20px;">
        <p>Made with ❤ using Streamlit</p>
    </footer>
""", unsafe_allow_html=True)
