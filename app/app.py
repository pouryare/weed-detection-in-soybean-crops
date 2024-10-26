import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Weed Detection in Soybean Crops",
    layout="centered"
)

# Constants
IMAGE_SIZE = 128
CLASS_NAMES = ['Broadleaf Weeds', 'Grass Weeds', 'Soil', 'Soybean Plants']

# Cache the model loading
@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        model_path = os.path.join(os.getcwd(), 'weed_detection.keras')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.error("Model file not found! Please ensure 'weed_detection.keras' is in the current directory.")
            return None

# Preprocess image
@st.cache_data
def preprocess_image(_image):
    """
    Preprocess image for model prediction
    Using _image parameter name to prevent Streamlit from hashing PIL.TiffImagePlugin.TiffImageFile objects
    """
    try:
        # Convert image to RGB if it's not
        if _image.mode != 'RGB':
            _image = _image.convert('RGB')
            
        # Resize image
        img = _image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Ensure array is float32
        img_array = img_array.astype(np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

# Make prediction
def predict_image(model, image):
    # Get prediction
    prediction = model.predict(image, verbose=0)
    return prediction[0]

# Function to display metrics
def display_metrics(predictions):
    cols = st.columns(4)
    max_prob_idx = np.argmax(predictions)
    
    for idx, (label, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        with cols[idx]:
            delta_color = "normal"
            if idx == max_prob_idx:
                delta_color = "off" if prob < 0.5 else "normal"
            
            st.metric(
                label=label,
                value=f"{prob:.1%}",
                delta=f"{prob-0.25:.1%}" if prob > 0.25 else f"-{0.25-prob:.1%}",
                delta_color=delta_color
            )

# Function to load and validate image
def load_image(uploaded_file):
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Get image info
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        img_width, img_height = image.size
        
        # Display image info
        st.write(f"""
        **Image Information:**
        - Size: {file_size:.2f} MB
        - Dimensions: {img_width}x{img_height}
        - Format: {image.format}
        - Mode: {image.mode}
        """)
        
        return image
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Main function
def main():
    st.title("Weed Detection in Soybean Crops")
    st.divider()
    
    # Load model at startup
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Description
    st.write("""
    This application uses a Convolutional Neural Network to classify different types of plants 
    and weeds in agricultural images. Upload an image to get started.
    
    **Supported formats:** JPG, JPEG, PNG, TIF, TIFF
    """)
    st.divider()
    
    # Image upload without form
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff']
    )
    
    if uploaded_file is not None:
        # Load and validate image
        image = load_image(uploaded_file)
        
        if image is not None:
            # Display original image
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
            st.divider()
            
            # Process image and make prediction
            with st.spinner("Analyzing image..."):
                # Add progress bar for visual feedback
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Preprocess and predict
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    predictions = predict_image(model, processed_image)
                    
                    # Display results
                    st.subheader("Classification Results")
                    display_metrics(predictions)
                    
                    # Show highest probability class
                    max_prob_idx = np.argmax(predictions)
                    max_prob = predictions[max_prob_idx]
                    
                    st.divider()
                    st.markdown(f"""
                    ### Primary Classification:
                    **{CLASS_NAMES[max_prob_idx]}** with **{max_prob:.1%}** confidence
                    """)
                    
                else:
                    st.error("Failed to process the image. Please try another image.")
            
    # Add information about the model
    st.divider()
    with st.expander("About the Model"):
        st.write("""
        This model was trained on a dataset of agricultural images containing:
        - Broadleaf weeds
        - Grass weeds
        - Soil
        - Soybean plants
        
        The model uses a CNN architecture with multiple convolutional layers and 
        achieves high accuracy in distinguishing between different types of vegetation 
        and soil in agricultural settings.
        
        **Image Requirements:**
        - Supported formats: JPG, JPEG, PNG, TIF, TIFF
        - Images will be automatically resized to 128x128 pixels
        - Color images are preferred, but grayscale images will be converted to RGB
        """)

if __name__ == "__main__":
    main()