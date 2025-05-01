import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="number_recommendation.tflite")
interpreter.allocate_tensors()

# Ambil info input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fungsi prediksi
def predict_digit(image_array):
    # Sesuaikan input shape
    image_array = image_array.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # tambah batch dimensi

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Jalankan inferensi
    interpreter.invoke()

    # Ambil hasil prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), output_data

# Streamlit UI
st.title("Prediksi Angka Tulisan Tangan")

uploaded_file = st.file_uploader("Upload gambar angka (28x28 grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))  # Grayscale dan resize
    img_array = np.array(img) / 255.0  # Normalisasi
    st.image(img, caption="Gambar diupload", width=150)

    if st.button("Prediksi"):
        label, confidence = predict_digit(img_array)
        st.write(f"Prediksi: **{label}**")
        st.write(f"Confidence: {confidence}")
