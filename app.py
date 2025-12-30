import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Malaria Detector (CNN)", page_icon="🔬", layout="wide")

st.title("🔬 Malaria Parasite Detection (CNN Model)")
st.markdown("""
This app uses a **Convolutional Neural Network (CNN)** to detect Malaria parasites.
It highlights the **Region of Interest** using Grad-CAM to show where the AI is looking.
""")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Make sure this matches the file you downloaded from Cell 3
    model_path = 'C:\\Users\\ADMIN\\Documents\\cnn\\final\\cnn_model.h5' 
    
    if not os.path.exists(model_path):
        st.error(f"❌ Error: '{model_path}' not found. Please download it from Colab Cell 3.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- GRAD-CAM FUNCTION (SIMPLIFIED FOR CNN) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv'):
    """
    Grad-CAM for a simple Sequential CNN.
    """
    # Create a sub-model that outputs the last conv layer and the prediction
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # We are interested in the class predicted by the model
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate Heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- MAIN APP UI ---
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload Cell Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    # 1. Load and Preprocess Image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize to 128x128 (Must match Colab IMG_SIZE)
    img_array = np.array(image.resize((128, 128)))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Display Original
    with col1:
        st.subheader("Microscopy Image")
        st.image(image, use_column_width=True)

    # 3. Analyze
    if st.sidebar.button("Run Diagnosis"):
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            # Thresholding (Adjust based on your training data)
            # Typically: < 0.5 = Parasitized, > 0.5 = Uninfected
            if score < 0.5:
                label = "Parasitized (Infected)"
                confidence = 1 - score
                color = "red"
            else:
                label = "Uninfected (Healthy)"
                confidence = score
                color = "green"
            
            st.markdown(f"### Prediction: :{color}[{label}]")
            st.write(f"**Confidence:** {confidence:.2%}")

            # 4. Explainability (Grad-CAM)
            try:
                # We specifically named the layer 'last_conv' in Cell 3
                heatmap = make_gradcam_heatmap(img_array, model, 'last_conv')
                
                # Resize heatmap to match original image size for display
                heatmap_resized = cv2.resize(heatmap, (128, 128))
                heatmap_resized = np.uint8(255 * heatmap_resized)
                heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Convert original image to Array/BGR for OpenCV
                original_img_np = np.array(image.resize((128, 128)))
                original_img_cv = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
                
                # Overlay
                overlay = cv2.addWeighted(original_img_cv, 0.6, heatmap_resized, 0.4, 0)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("AI Focus Area (Grad-CAM)")
                    st.image(overlay, caption="Red areas = Parasite features detected", use_column_width=True)
                    
            except Exception as e:
                st.warning(f"Could not generate XAI. (Error: {e})")
                st.info("Tip: Ensure the layer name 'last_conv' exists in the model.")

elif model is None:
    st.info("⚠️ Please ensure 'cnn_model.h5' is in the folder.")
else:
    st.info("👈 Upload an image to start.")