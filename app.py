import streamlit as st
import os
import random
import matplotlib.pyplot as plt
from helper import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, Sequential

# App configuration
st.set_page_config(page_title="Deepfake Classifier", layout="wide")

st.title("Deepfake Image Classifier")

# Load model
model = load_cnn_model()

# Dataset directory
# MAIN = "./Dataset"
# train_dir = os.path.join(MAIN, "Train")
# val_dir = os.path.join(MAIN, "Validation")
# test_dir = os.path.join(MAIN, "Test")
DATASET_PATH = "./Trimmed Dataset"
train_dir = os.path.join(DATASET_PATH, "Train")
val_dir = os.path.join(DATASET_PATH, "Validation")
test_dir = os.path.join(DATASET_PATH, "Test")


# Data pipeline
def get_dataset(dir_path):
    dataset = image_dataset_from_directory(
        dir_path,
        image_size=(128, 128),
        batch_size=32,
        label_mode='binary'
    )
    return dataset.map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=tf.data.AUTOTUNE)


def count_images_in_dir(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

train_count = count_images_in_dir(train_dir)
val_count = count_images_in_dir(val_dir)
test_count = count_images_in_dir(test_dir)


# Cached datasets
@st.cache_resource
def load_datasets():
    return get_dataset(train_dir), get_dataset(val_dir), get_dataset(test_dir)

train_ds, val_ds, test_ds = load_datasets()

# Tabs
tabs = st.tabs(["📁 Dataset Viewer", "📊 Model Evaluation", "🧪 Classified Samples", "🖼️ Upload and Predict"])

# --- Dataset Viewer ---
with tabs[0]:
    st.title("📁 Explore Dataset")

     # Show stats in 3 columns
    col1, col2, col3 = st.columns(3)
    col1.metric("🟦 Train Images", train_count)
    col2.metric("🟨 Validation Images", val_count)
    col3.metric("🟥 Test Images", test_count)


    category = st.selectbox("Choose Dataset", ["Train", "Validation", "Test"])
    label = st.radio("Class", ["Real", "Fake"])
    sample_dir = os.path.join(DATASET_PATH, category, label)

    images = get_sample_images(sample_dir, 5)
    st.subheader(f"Sample {label} Images from {category}")

    cols = st.columns(len(images))
    for col, img_path in zip(cols, images):
        col.image(img_path, use_container_width=True, caption=os.path.basename(img_path))


# --- Model Evaluation ---
with tabs[1]:
    st.title("📊 Model Evaluation on Test Set")

    results_file = "evaluation_results.npz"

    if st.button("Evaluate Model"):
        with st.spinner("Evaluating model or loading saved results..."):
            try:
                # Try loading existing evaluation results
                data = np.load(results_file)
                y_true = data["y_true"]
                y_pred = data["y_pred"]
                st.info("Loaded evaluation results from saved file.")
            except FileNotFoundError:
                # Perform real-time evaluation and save results
                y_true, y_pred = evaluate_model(model, test_ds)
                np.savez(results_file, y_true=y_true, y_pred=y_pred)
                st.success("Evaluation complete. Results saved for future use.")

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_confusion_matrix(y_true, y_pred)
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.subheader("Classification Report")
            st.code(get_classification_report(y_true, y_pred))


# --- Classified Samples ---
with tabs[2]:
    st.title("🧪 Classified Sample Predictions")

    sample_batch = next(iter(test_ds))
    images, labels = sample_batch
    preds = (model.predict(images).flatten() > 0.5).astype(int)

    actual_batch_size = len(images)
    num_samples = min(10, actual_batch_size) 
    num_cols = 5

    for row_idx in range(0, num_samples, num_cols):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            sample_idx = row_idx + col_idx
            if sample_idx >= num_samples:
                break

            img = images[sample_idx].numpy()
            actual = "Real" if labels[sample_idx].numpy() == 1 else "Fake"
            predicted = "Real" if preds[sample_idx] == 1 else "Fake"
            cols[col_idx].image(img, caption=f"Actual: {actual} | Predicted: {predicted}", width=150)

# --- Upload and Predict ---
with tabs[3]:
    st.title("🖼️ Upload an Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        from PIL import Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=256)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image..."):
                label, confidence, _ = predict_image(model, uploaded_file)
                color = 'red' if label.lower() == 'fake' else 'green'

                st.markdown(
                    f"<div style='text-align: center;'><h4>Prediction: <span style='color: {color};'>{label}</span> ({confidence*100:.2f}%)</h4></div>",
                    unsafe_allow_html=True
                )

