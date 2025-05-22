import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tempfile
import os
import gdown



# Title and description
st.title("🍕 Pizza Classifier")
st.write("Upload an image to check if it's a pizza!")

# Load model only once
@st.cache_resource
def load_pizza_model(model_path='best_pizza_model_vgg19.keras'):
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1MOpkyA88FLwpMl90FNWU8wXxqILCmWQA"
        gdown.download(url, model_path, quiet=False)
    model = load_model(model_path)
    return model

model = load_pizza_model()
class_names = ['not_pizza', 'pizza']
img_size = (224, 224)
# img_show_size = (512, 512)

# Upload file
uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Show the uploaded image
    image = Image.open(uploaded)
    # image = cv2.resize(image, img_show_size)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, img_size)
    img = img / 255.0

    # Predict
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    index = int(np.argmax(pred, axis=0))
    label = class_names[index]
    accuracy = f"{float(pred[index]):.4f}"

    # Show results
    st.subheader(f"Prediction: **{label}** **({accuracy})**")
    
    # st.write(f"# debug:")
    # st.write(pred)
