import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# 1. Configuration
st.set_page_config(
    page_title="Malaria Detection AI", 
    page_icon="mn", 
    layout="wide"
)

# 2. Load the Model (Cached for performance)
@st.cache_resource
def load_model():
    model_path = 'C:\\Users\\ADMIN\\Documents\\cnn\\final\\malaria_model.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Could not find '{model_path}'. Please make sure it is in the same folder as app.py.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# 3. Grad-CAM Visualization Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    """
    Generates a heatmap showing where the model is looking.
    Compatible with Functional API models.
    """
    # 1. Find the VGG16 layer within the model
    # In Functional API, layers are: [Input, VGG16, GlobalPool, Dense...]
    # We search for the layer named 'vgg16' or assume it's index 1
    vgg_base = None
    for layer in model.layers:
        if 'vgg16' in layer.name:
            vgg_base = layer
            break
    
    # Fallback if name finding fails (usually it's layer index 1)
    if vgg_base is None:
        vgg_base = model.layers[1]

    # 2. Create the Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[vgg_base.inputs],
        outputs=[vgg_base.get_layer(last_conv_layer_name).output, vgg_base.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, _ = grad_model(img_array)
        loss = tf.reduce_mean(last_conv_layer_output)

    grads = tape.gradient(loss, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 4. Main UI Layout
st.title("🔬 Malaria Parasite Detection & XAI")
st.markdown("Upload a **Microscopy Cell Image** to detect if it is infected.")

# Sidebar for Upload
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Load Model
model = load_model()

if uploaded_file is not None and model is not None:
    # -- Layout: Two Columns --
    col1, col2 = st.columns(2)
    
    # -- 1. Process Image --
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize to 128x128 (Model Requirement)
    img_array = np.array(image.resize((128, 128)))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -- 2. Display Original --
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # -- 3. Make Prediction on Button Click --
    if st.sidebar.button("Analyze Image"):
        with st.spinner('Running AI Analysis...'):
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            # Interpret Result (Assuming 0=Parasitized, 1=Uninfected based on TFDS standard)
            # Note: Adjust logic if your specific training run flipped the labels.
            # Usually: Close to 0 is Parasitized, Close to 1 is Uninfected.
            if score < 0.5:
                label = "Parasitized (Infected)"
                confidence = 1 - score
                color = "red"
            else:
                label = "Uninfected (Healthy)"
                confidence = score
                color = "green"

            # Display Text Result
            st.success("Analysis Complete!")
            st.markdown(f"### Prediction: :{color}[{label}]")
            st.write(f"**Confidence Score:** {confidence:.2%}")

            # -- 4. Generate Explainable AI (Grad-CAM) --
            try:
                heatmap = make_gradcam_heatmap(img_array, model)

                # Process Heatmap for Overlay
                heatmap_resized = cv2.resize(heatmap, (128, 128))
                heatmap_resized = np.uint8(255 * heatmap_resized)
                heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Convert Original to format for merging
                original_img_np = np.array(image.resize((128, 128)))
                original_img_cv = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

                # Overlay
                overlay = cv2.addWeighted(original_img_cv, 0.6, heatmap_resized, 0.4, 0)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                # Display XAI
                with col2:
                    st.subheader("AI 'Vision' (Grad-CAM)")
                    st.image(overlay, use_column_width=True, caption="Red/Yellow = Areas the AI focused on")
                    
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")

elif model is None:
    st.warning("Waiting for model file...")
else:
    st.info("👈 Please upload an image using the sidebar to start.")