import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from streamlit_lottie import st_lottie
import json
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model


# -------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(layout="wide")


# -------------------------------------------------
# LOTTIE LOADER
# -------------------------------------------------
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

lottie_loading = load_lottie("loading.json")


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_models():

    histopath_model_path = hf_hub_download(
        repo_id="Adharsh102/cancer-detection-models",
        filename="resnet50_cancer_model-finetuned-version-1.keras"
    )

    MRI_model_path = hf_hub_download(
        repo_id="Adharsh102/cancer-detection-models",
        filename="resnet50_cancer_model-MRI-finetuned-version-1.keras"
    )

    histopathological_model = load_model(histopath_model_path, compile=False)
    MRI_model = load_model(MRI_model_path, compile=False)

    return histopathological_model, MRI_model
histopathological_model, MRI_model = load_models()
# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------
histo_class_names = [
    'Acute Lymphoblastic Leukemia_early',
    'Acute Lymphoblastic Leukemia_normal',
    'Acute Lymphoblastic Leukemia_pre',
    'Acute Lymphoblastic Leukemia_pro',
    'breast_malignant',
    'breast_normal',
    'lung_colon_Adenocarcinoma',
    'lung_colon_normal',
    'lung_squamous cell carcinoma'
]

MRI_class_names = [
    'brain_glioma_tumor',
    'brain_meningioma_tumor',
    'brain_normal',
    'brain_pituitary_tumor',
    'kidney_cyst',
    'kidney_normal',
    'kidney_stone',
    'kidney_tumor',
    'pancreatic_normal',
    'pancreatic_tumor'
]


# -------------------------------------------------
# GRADCAM
# -------------------------------------------------
def generate_gradcam(model, img_array, img_path, class_index, class_names, confidence,
                     layer_name='conv5_block3_out'):

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = 1 - heatmap_resized
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)

    combined_img = np.hstack((original_rgb, superimposed_img))

    st.image(
        combined_img,
        caption=f"Grad-CAM: {class_names[class_index]} (Confidence: {confidence:.2f})",
        use_container_width=True
    )


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Cancer Classifier with CNN ResNet50 and Grad-CAM")
st.divider()


# -------------------------------------------------
# EXAMPLE IMAGES SECTION
# -------------------------------------------------
st.header("Example Images")

cols = st.columns(6, gap="medium")

image_paths = [
    "examples/image1.jpg",
    "examples/image2.jpg",
    "examples/image3.jpg",
    "examples/image4.jpg",
    "examples/image5.jpg",
    "examples/image6.jpg"
]

for col, img in zip(cols, image_paths):
    with col:
        st.image(img, use_container_width=True)

st.divider()


# -------------------------------------------------
# MODEL SELECTION (RADIO BUTTON)
# -------------------------------------------------


with st.container():

    st.subheader("Select Image Type")

    model_choice = st.radio(
        "",
        ["Histopathological", "MRI"],
        horizontal=True
    )

st.divider()


left, center, right = st.columns([1, 2, 1])

with center:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

# PREDICTION
# -------------------------------------------------
if uploaded_file:
    
    img_path = f"./temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    loader = st.empty()

    # show animation
    with loader.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st_lottie(lottie_loading, height=220)

    # -------------
    # Manual routing
    # -------------
    if model_choice == "Histopathological":
        left, center, right = st.columns([1, 2, 1])

        result = histopathological_model.predict(img_array)
        idx = np.argmax(result, axis=1)[0]
        conf = result[0][idx]

        loader.empty()
        with center:
            generate_gradcam(
            histopathological_model,
            img_array,
            img_path,
            idx,
            histo_class_names,
            conf
        )

    else:
        left, center, right = st.columns([1, 2, 1])

        result = MRI_model.predict(img_array)
        idx = np.argmax(result, axis=1)[0]
        conf = result[0][idx]

        loader.empty()
        with center:
            generate_gradcam(
            MRI_model,
            img_array,
            img_path,
            idx,
            MRI_class_names,
            conf
        )
