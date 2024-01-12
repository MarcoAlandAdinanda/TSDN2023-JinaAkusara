import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

CLASS_NAMES = ["Benign", "Malignant"]

# Load the model
model = tf.keras.models.load_model('Jina_Akusara_TSDN_2023_FIX.h5')

# Predict the Image
def classify_image(img):
    # Preprocess the image before feeding it to the model
    img_resized= cv2.resize(img, (512,512))
    img = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img)
    output_class=CLASS_NAMES[np.argmax(prediction)]
    return output_class

# Create Gradio interface
gr.Interface(fn=classify_image, inputs="image", outputs="text", title="ANDROID MALWARE DETECTION USING NEURAL NETWORKS | Jina Akusara - Universitas Gadjah Mada").launch()
