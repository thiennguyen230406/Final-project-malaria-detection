import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# 1. Cấu hình trang
st.set_page_config(
    page_title="Malaria Detector (CNN)", 
    page_icon="🔬", 
    layout="wide"
)

# 2. Tải Model (CNN)
@st.cache_resource
def load_model():
    model_path = 'C:\\Users\\ADMIN\\Documents\\cnn\\final\\cnn_model.h5' 
    
    if not os.path.exists(model_path):
        st.error(f"❌ Lỗi: Không tìm thấy file '{model_path}'.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Lỗi tải model: {e}")
        return None

# 3. Hàm Grad-CAM (ĐÃ CHỈNH SỬA ĐỂ CNN TRÔNG GIỐNG VGG)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Fixed Grad-CAM function that rebuilds the graph to avoid
    'AttributeError: The layer has never been called' in Keras 3.
    """

    # 1. Create a dummy input tensor with the same shape as your images
    # Note: We use the shape from the img_array provided (batch_size, 128, 128, 3)
    input_shape = img_array.shape[1:] # Returns (128, 128, 3)
    inputs = tf.keras.Input(shape=input_shape)

    # 2. Re-trace the model graph using the *existing* trained layers
    # This bypasses the "never been called" error by calling them explicitly now.
    x = inputs
    last_conv_output = None

    for layer in model.layers:
        x = layer(x)
        # Check if this is the layer we want to watch
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    # Final output of the model (prediction)
    model_output = x

    # 3. Create the Grad-CAM model
    # Now we have a clear path from 'inputs' to 'last_conv_output' and 'model_output'
    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[last_conv_output, model_output])

    # 4. Compute Gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Determine which class we are explaining
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Calculate gradient of the predicted class w.r.t. the feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool the gradients (average over the height/width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply the feature map by the pooled gradients (importance)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

# 4. Giao diện chính
st.title("🔬 Malaria Parasite Detection")
st.markdown("---") 

# Sidebar
st.sidebar.header("Input Image")
uploaded_file = st.sidebar.file_uploader("Upload file here", type=["jpg", "png", "jpeg"])

model = load_model()

if uploaded_file is not None and model is not None:
    # Xử lý ảnh
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image.resize((128, 128)))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Nút chạy
    if st.sidebar.button("Analyze Image"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner('AI is thinking...'):
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            if score < 0.5:
                label = "Parasitized (Infected)"
                confidence = 1 - score
                color = "red"
            else:
                label = "Uninfected (Healthy)"
                confidence = score
                color = "green"

            st.sidebar.markdown(f"### Prediction: :{color}[{label}]")
            st.sidebar.write(f"Confidence: **{confidence:.2%}")

            # Tạo Grad-CAM
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv')

                # Resize heatmap về 128x128
                heatmap_resized = cv2.resize(heatmap, (128, 128))
                
                # Chuyển sang thang màu JET
                heatmap_resized = np.uint8(255 * heatmap_resized)
                heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Ảnh gốc
                original_img_np = np.array(image.resize((128, 128)))
                original_img_cv = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

                # TRỘN ẢNH (Overlay)
                # Tăng Heatmap lên 0.6, giảm ảnh gốc xuống 0.4 để màu rực rỡ hơn
                overlay = cv2.addWeighted(original_img_cv, 0.4, heatmap_resized, 0.6, 0)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                with col2:
                    st.image(overlay, caption="AI Focus Area (Grad-CAM)", use_container_width=True)
                    st.caption("Red/Yellow = Areas the AI focused on")
                    
            except Exception as e:
                st.warning(f"Could not generate XAI: {e}")

elif model is None:
    st.warning("Waiting for model...")
else:
    st.info("👈 Upload an image to start.")