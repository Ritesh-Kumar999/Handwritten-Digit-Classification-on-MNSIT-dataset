import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load your trained model
model = joblib.load('Handwritten')

def preprocess_image(image):
    # Convert to grayscale if the image is in RGB mode
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize the image to 28x28
    image_resized = image.resize((28, 28))
    
    # Normalize pixel values by dividing by 255
    image_normalized = np.array(image_resized) / 255.0
    
    # Reshape the image to (1, 28, 28) for prediction
    image_reshaped = np.reshape(image_normalized, (1, 28, 28))
    
    return image_reshaped

st.title('Handwritten Digit Classifier')

uploaded_file = st.file_uploader('Upload a PNG image', type=['png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the digit
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    st.write(f'Predicted Digit: {predicted_digit}')
