import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

IMAGE_SIZE = (128, 128)

def load_cnn_model(model_path='fake_real_image_classifier_cnn.h5'):
    return load_model(model_path)

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32), img

def predict_image(model, image_file):
    img_array, img = preprocess_image(image_file)
    prediction = model.predict(img_array)[0][0]
    label = "Real" if prediction > 0.5 else "Fake"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence, img

def get_sample_images(folder_path, num_images=5):
    image_files = sorted(os.listdir(folder_path))[:num_images]
    return [os.path.join(folder_path, img) for img in image_files]

def evaluate_model(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = (model.predict(images).flatten() > 0.5).astype(int)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3)) 
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()  
    return fig

def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=["Fake", "Real"], output_dict=False)
