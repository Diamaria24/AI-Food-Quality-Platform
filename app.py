import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_food_model():

    model = load_model(
        "models/trained/mobilenet.keras"
    )

    return model


model = load_food_model()

# =========================================================
# Classes
# =========================================================

classes = [

    "Apple Fresh",
    "Apple Rotten",
    "Banana Fresh",
    "Banana Rotten",
    "Strawberry Fresh",
    "Strawberry Rotten",
    "Tomato Fresh",
    "Tomato Rotten"

]

# =========================================================
# UI
# =========================================================

st.set_page_config(

    page_title="AI Food Quality Platform",

    page_icon="🍎",

    layout="centered"
)

st.title("🍎 AI Food Quality Platform")

st.markdown(
    "Upload a food image to classify its quality."
)

uploaded_file = st.file_uploader(

    "Choose an Image",

    type=["jpg", "jpeg", "png"]

)

# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    # Convert RGBA → RGB
    image = image.convert(
        "RGB"
    )

    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True
    )

    # -----------------------------------------

    img = image.resize(
        (224, 224)
    )

    img = np.array(
        img
    ).astype(
        np.float32
    )

    img = np.expand_dims(

        img,

        axis=0
    )

    img = preprocess_input(
        img
    )

    # -----------------------------------------

    prediction = model.predict(

        img,

        verbose=0
    )

    idx = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    predicted_class = classes[idx]

    # -----------------------------------------

    st.success(

        f"Prediction : {predicted_class}"
    )

    st.write(

        f"Confidence : {confidence*100:.2f}%"
    )

    st.progress(
        float(confidence)
    )

    # -----------------------------------------

    st.subheader(
        "Class Probabilities"
    )

    for i, cls in enumerate(classes):

        st.write(

            f"{cls} : "

            f"{prediction[0][i]*100:.2f}%"
        )