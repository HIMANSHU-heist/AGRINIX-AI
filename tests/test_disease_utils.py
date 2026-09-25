import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from image_utils import preprocess_leaf_image, predict_disease


def make_fake_uploaded_file():
    """Build an in-memory JPEG that behaves like a Streamlit UploadedFile for read()/seek()."""
    img = Image.new("RGB", (300, 300), color=(60, 140, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_preprocess_leaf_image_shape():
    fake_file = make_fake_uploaded_file()
    result = preprocess_leaf_image(fake_file, img_size=224)
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32


class FakeModel:
    """Minimal stand-in for a Keras model so predict_disease can be unit tested without TensorFlow."""
    def predict(self, x):
        # Pretend class index 2 ("Potato___healthy") wins with high confidence
        return np.array([[0.02, 0.03, 0.90, 0.05]])


def test_predict_disease_returns_label_and_info():
    labels = ["Pepper__bell___healthy", "Potato___Early_blight", "Potato___healthy", "Tomato_healthy"]
    disease_info = {
        "Potato___healthy": {"severity": "None", "action": "No treatment needed."}
    }
    fake_file = make_fake_uploaded_file()
    label, confidence, info = predict_disease(fake_file, FakeModel(), labels, disease_info)

    assert label == "Potato___healthy"
    assert confidence == 90.0
    assert info["severity"] == "None"


def test_predict_disease_unknown_label_fallback():
    labels = ["ClassA", "ClassB"]
    disease_info = {}  # no info for either class

    class TwoClassModel:
        def predict(self, x):
            return np.array([[0.1, 0.9]])

    fake_file = make_fake_uploaded_file()
    label, confidence, info = predict_disease(fake_file, TwoClassModel(), labels, disease_info)
    assert label == "ClassB"
    assert info["severity"] == "Unknown"
