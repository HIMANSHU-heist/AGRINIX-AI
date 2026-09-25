"""
Pure image-preprocessing helpers used by the disease-detection tab.
Kept separate from app.py (which does Streamlit page setup at import time)
so this module can be unit-tested in CI without a Streamlit runtime.
"""

import cv2
import numpy as np


def preprocess_leaf_image(uploaded_file, img_size=224):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


def predict_disease(uploaded_file, model, labels, disease_info):
    img_array = preprocess_leaf_image(uploaded_file)
    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    label = labels[top_idx]
    confidence = float(preds[top_idx]) * 100
    info = disease_info.get(
        label,
        {"severity": "Unknown", "action": "Consult a local agriculture officer for exact treatment."},
    )
    return label, confidence, info
