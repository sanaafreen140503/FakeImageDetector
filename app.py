
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request

# ----------------------------------
# 1. Page Configuration
# ----------------------------------
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

# ----------------------------------
# 2. Model Configuration
# ----------------------------------
MODEL_URL = (
    "https://huggingface.co/ukiiii/fake-image-detector-model/"
    "resolve/main/hybrid_cnn_resnet50.keras"
)

MODEL_PATH = Path(__file__).parent / "hybrid_cnn_resnet50.keras"


# ----------------------------------
# 3. Download Model if Missing
# ----------------------------------
def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    try:
        MODEL_PATH.unlink(missing_ok=True)

        with st.spinner("Downloading AI detection model..."):
            request = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(
                request, timeout=300
            ) as response:
                with open(MODEL_PATH, "wb") as model_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        model_file.write(chunk)

        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            raise RuntimeError("The downloaded model file is empty.")

    except Exception as e:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Model download failed. Check the Hugging Face model URL. "
            f"Details: {e}"
        ) from e


# ----------------------------------
# 4. Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    download_model()

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
        safe_mode=False
    )


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the AI detection model.")
    st.exception(e)
    st.stop()


# ----------------------------------
# 5. Upload Image
# ----------------------------------
st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image to analyze",
    type=["jpg", "jpeg", "png", "webp"]
)


# ----------------------------------
# 6. Analyze Image
# ----------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔍 Analyze Image", type="primary"):

            with st.spinner("Analyzing image..."):

                # Resize to the model's expected input size
                image_array = np.array(image, dtype=np.float32)

                image_tensor = tf.convert_to_tensor(image_array)

                image_tensor = tf.image.resize(
                    image_tensor,
                    (32, 32)
                )

                # Normalize pixel values to [0, 1]
                image_tensor = image_tensor / 255.0

                # Add batch dimension
                image_tensor = tf.expand_dims(
                    image_tensor,
                    axis=0
                )

                # Generate prediction
                prediction = model.predict(
                    image_tensor,
                    verbose=0
                )

                # Assumes sigmoid output: 1 = REAL, 0 = FAKE
                probability_real = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

                probability_real = float(
                    np.clip(probability_real, 0.0, 1.0)
                )

                probability_fake = 1.0 - probability_real

                if probability_real >= 0.5:
                    label = "REAL"
                    score = probability_real
                else:
                    label = "FAKE"
                    score = probability_fake

                # Display result
                st.subheader("Analysis Result")

                if label == "REAL":
                    st.success("✅ Prediction: REAL")
                else:
                    st.warning("⚠️ Prediction: FAKE")

                st.metric(
                    "Model score for predicted class",
                    f"{score * 100:.2f}%"
                )

                st.progress(score)

                # Display both scores
                with st.expander("View prediction scores"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "REAL score",
                            f"{probability_real * 100:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "FAKE score",
                            f"{probability_fake * 100:.2f}%"
                        )

                st.caption(
                    "These scores are model outputs, not proof of "
                    "authenticity. The model may make mistakes, "
                    "especially on unfamiliar images."
                )

    except Exception as e:
        st.error("Unable to analyze this image.")
        st.exception(e)


# ----------------------------------
# 7. About This Project
# ----------------------------------
with st.expander("About this project"):
    st.markdown(
        """
        **Model:** Hybrid CNN + ResNet50

        **Dataset:** CIFAKE

        **Reported test accuracy:** 96.74%

        The hybrid model combines features from a custom CNN
        and a ResNet50 feature extractor.
        """
    )
