import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("pneumonia_model.h5", compile=False)

st.title("🫁 Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("Prediction: Pneumonia")
    else:
        st.success("Prediction: Normal")

    st.write("Confidence:", float(prediction))
