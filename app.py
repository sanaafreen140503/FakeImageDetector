
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Fake Image Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Fake Image Detector")
st.write(
    "Detect whether an image is likely to be AI-generated "
    "or real using a Hybrid CNN + ResNet50 model."
)

st.info(
    "The model achieved 96.74% test accuracy on CIFAKE. "
    "Performance may differ on unfamiliar images."
)

# -----------------------------
# 2. Load the trained model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "hybrid_cnn_resnet50.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Ensure "
        "'hybrid_cnn_resnet50.keras' is in the repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

# -----------------------------
# 3. Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png", "webp"]
)

# -----------------------------
# 4. Predict image class
# -----------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):

                # The model expects 32 x 32 x 3 input
                image_array = np.array(image)
                image_tensor = tf.convert_to_tensor(image_array)
                image_tensor = tf.image.resize(
                    image_tensor, (32, 32)
                )
                image_tensor = tf.expand_dims(
                    image_tensor, axis=0
                )

                # Output: sigmoid probability for REAL (class 1)
                prediction = model.predict(
                    image_tensor, verbose=0
                )

                probability_real = float(prediction[0][0])
                probability_real = np.clip(
                    probability_real, 0.0, 1.0
                )

                if probability_real >= 0.5:
                    label = "REAL"
                    confidence = probability_real
                else:
                    label = "FAKE"
                    confidence = 1.0 - probability_real

                st.subheader("Analysis Result")

                if label == "REAL":
                    st.success("Prediction: REAL")
                else:
                    st.warning("Prediction: FAKE")

                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(float(confidence))

                with st.expander("View prediction scores"):
                    st.write(
                        f"FAKE score: "
                        f"{(1 - probability_real) * 100:.2f}%"
                    )
                    st.write(
                        f"REAL score: "
                        f"{probability_real * 100:.2f}%"
                    )

                st.caption(
                    "This is a model prediction, not proof of "
                    "an image's authenticity. Confidence is not "
                    "a guarantee of correctness."
                )

    except Exception as e:
        st.error(f"Unable to analyze this image: {e}")

# -----------------------------
# 5. About the project
# -----------------------------
with st.expander("About this project"):
    st.markdown(
        """
        **Model:** Hybrid CNN + ResNet50

        **Dataset:** CIFAKE

        **Reported test accuracy:** 96.74%

        The hybrid model combines features from a custom CNN
        and a pretrained ResNet50 feature extractor.
        """
    )
