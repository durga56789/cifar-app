# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:14:01 2025

@author: ariji
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import io
import base64
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 1.1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class CIFAR10Predictor:
    """
    A class to handle CIFAR-10 model loading and predictions
    """
    
    def __init__(self):
        self.model = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.model_loaded = False
        
    @st.cache_resource
    def load_model_from_files(_self, model_json_path, model_weights_path):
        """
        Load model from JSON architecture and H5 weights files
        """
        try:
            # Load model architecture
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            # Create model from JSON
            model = model_from_json(loaded_model_json)
            
            # Load weights
            model.load_weights(model_weights_path)
            
            _self.model = model
            _self.model_loaded = True
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    @st.cache_resource
    def load_model_h5(_self, model_path):
        """
        Load complete model from H5 file
        """
        try:
            model = load_model(model_path)
            _self.model = model
            _self.model_loaded = True
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def preprocess_image(self, image, target_size=(32, 32)):
        """
        Preprocess image for CIFAR-10 prediction
        """
        # Resize image to 32x32
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Ensure 3 channels (RGB)
        if len(image_resized.shape) == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        elif image_resized.shape[2] == 4:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGBA2RGB)
        
        # Convert to float32 and normalize
        image_normalized = image_resized.astype('float32')
        
        # Apply same normalization as training (you may need to adjust these values)
        # Based on your original code, you used mean and std from training data
        # For CIFAR-10, common normalization values are:
        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([63.0, 62.1, 66.7])
        
        image_normalized = (image_normalized - mean) / std
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_resized
    
    def predict_image(self, image):
        """
        Make prediction on a single image
        """
        if not self.model_loaded:
            return None, None, None
        
        # Preprocess image
        processed_image, display_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images
        """
        if not self.model_loaded:
            return None
        
        batch_predictions = []
        for image in images:
            processed_image, _ = self.preprocess_image(image)
            prediction = self.model.predict(processed_image, verbose=0)
            batch_predictions.append(prediction[0])
        
        return np.array(batch_predictions)

def save_to_database(predictions_df, user, password, database, table_name='cifar10_predictions'):
    """
    Save predictions to MySQL database
    """
    try:
        # Create database connection
        engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{database}")
        
        # Add timestamp
        predictions_df['timestamp'] = datetime.now()
        
        # Save to database
        predictions_df.to_sql(table_name, con=engine, if_exists='append', index=False)
        
        return True, "Predictions saved to database successfully!"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def create_prediction_chart(predictions, class_names):
    """
    Create an interactive bar chart of predictions
    """
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': predictions
    })
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, x='Probability', y='Class', orientation='h',
                 title='Class Prediction Probabilities',
                 color='Probability',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Probability",
        yaxis_title="Class"
    )
    
    return fig

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🖼️ CIFAR-10 Image Classifier</h1>
            <p>Upload images or use your camera to classify them into 10 categories</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = CIFAR10Predictor()
    
    # Sidebar for model loading and configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # Model loading options
    model_load_option = st.sidebar.selectbox(
        "Choose model loading method:",
        ["Upload Model Files", "Use Local Files"]
    )
    
    if model_load_option == "Upload Model Files":
        st.sidebar.subheader("Upload Model Files")
        
        # File uploaders for model files
        json_file = st.sidebar.file_uploader("Upload model architecture (JSON)", type=['json'])
        weights_file = st.sidebar.file_uploader("Upload model weights (H5)", type=['h5'])
        
        if json_file is not None and weights_file is not None:
            # Save uploaded files temporarily
            with open("temp_model.json", "wb") as f:
                f.write(json_file.getbuffer())
            with open("temp_weights.h5", "wb") as f:
                f.write(weights_file.getbuffer())
            
            # Load model
            model = predictor.load_model_from_files("temp_model.json", "temp_weights.h5")
            
            if model is not None:
                st.sidebar.success("✅ Model loaded successfully!")
            
    else:
        st.sidebar.subheader("Local Model Files")
        
        model_path_option = st.sidebar.selectbox(
            "Model file type:",
            ["Complete Model (.h5)", "Architecture + Weights (.json + .h5)"]
        )
        
        if model_path_option == "Complete Model (.h5)":
            model_path = st.sidebar.text_input("Model path (.h5):", "model.h5")
            if st.sidebar.button("Load Model") and model_path:
                if os.path.exists(model_path):
                    model = predictor.load_model_h5(model_path)
                    if model is not None:
                        st.sidebar.success("✅ Model loaded successfully!")
                else:
                    st.sidebar.error("Model file not found!")
        
        else:
            json_path = st.sidebar.text_input("JSON architecture path:", "model.json")
            weights_path = st.sidebar.text_input("H5 weights path:", "model.h5")
            
            if st.sidebar.button("Load Model") and json_path and weights_path:
                if os.path.exists(json_path) and os.path.exists(weights_path):
                    model = predictor.load_model_from_files(json_path, weights_path)
                    if model is not None:
                        st.sidebar.success("✅ Model loaded successfully!")
                else:
                    st.sidebar.error("Model files not found!")
    
    # Database configuration
    st.sidebar.header("🗄️ Database Configuration")
    save_to_db = st.sidebar.checkbox("Save predictions to database")
    
    if save_to_db:
        db_user = st.sidebar.text_input("Database User", "root")
        db_password = st.sidebar.text_input("Database Password", type="password")
        db_name = st.sidebar.text_input("Database Name", "cifar10_db")
        table_name = st.sidebar.text_input("Table Name", "predictions")
    
    # Main content area
    if not predictor.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar configuration.")
        st.info("""
        **Instructions:**
        1. Choose your model loading method in the sidebar
        2. Provide the model files (JSON + H5 or complete H5)
        3. Load the model
        4. Start making predictions!
        """)
        return
    
    # Prediction interface tabs
    tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Batch Images", "📊 Model Info"])
    
    with tab1:
        st.header("Single Image Prediction")
        
        # Image input options
        input_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Camera Capture", "Sample Images"]
        )
        
        uploaded_image = None
        
        if input_option == "Upload Image":
            uploaded_image = st.file_uploader(
                "Choose an image...", 
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
            )
        
        elif input_option == "Camera Capture":
            uploaded_image = st.camera_input("Take a picture")
        
        elif input_option == "Sample Images":
            sample_images = {
                "Airplane": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Airplane_silhouette.svg/256px-Airplane_silhouette.svg.png",
                "Car": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Red_car_icon.svg/256px-Red_car_icon.svg.png",
                "Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/256px-Cat03.jpg"
            }
            
            selected_sample = st.selectbox("Choose a sample image:", list(sample_images.keys()))
            if st.button("Load Sample Image"):
                st.info("Sample image feature requires internet connection and proper URL handling.")
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    predicted_class, confidence, all_predictions = predictor.predict_image(image)
                
                if predicted_class is not None:
                    # Display prediction
                    st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🎯 Predicted Class: {predicted_class}</h3>
                            <h4>🎲 Confidence: {confidence:.2%}</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create and display prediction chart
                    fig = create_prediction_chart(all_predictions, predictor.class_names)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to database if enabled
                    if save_to_db and db_user and db_password and db_name:
                        prediction_data = pd.DataFrame({
                            'predicted_class': [predicted_class],
                            'confidence': [confidence],
                            'image_name': [uploaded_image.name if hasattr(uploaded_image, 'name') else 'camera_capture']
                        })
                        
                        success, message = save_to_database(prediction_data, db_user, db_password, db_name, table_name)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
    
    with tab2:
        st.header("Batch Image Prediction")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images...", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("Predict All Images", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    predicted_class, confidence, _ = predictor.predict_image(image)
                    
                    results.append({
                        'Image': uploaded_file.name,
                        'Predicted Class': predicted_class,
                        'Confidence': f"{confidence:.2%}"
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("Batch Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Save batch results to database
                if save_to_db and db_user and db_password and db_name:
                    batch_data = pd.DataFrame({
                        'predicted_class': [r['Predicted Class'] for r in results],
                        'confidence': [float(r['Confidence'].strip('%'))/100 for r in results],
                        'image_name': [r['Image'] for r in results]
                    })
                    
                    success, message = save_to_database(batch_data, db_user, db_password, db_name, table_name)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    with tab3:
        st.header("Model Information")
        
        if predictor.model_loaded:
            # Model summary
            st.subheader("Model Architecture")
            
            # Create a string buffer to capture model summary
            stringlist = []
            predictor.model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            
            st.text(model_summary)
            
            # Model metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <h4>Total Parameters</h4>
                        <h2>{:,}</h2>
                    </div>
                """.format(predictor.model.count_params()), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <h4>Model Layers</h4>
                        <h2>{}</h2>
                    </div>
                """.format(len(predictor.model.layers)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="metric-card">
                        <h4>Classes</h4>
                        <h2>10</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # Class information
            st.subheader("CIFAR-10 Classes")
            class_info = pd.DataFrame({
                'Index': range(10),
                'Class Name': predictor.class_names,
                'Description': [
                    'Fixed-wing aircraft', 'Four-wheeled motor vehicle', 'Flying animal',
                    'Domestic feline', 'Hoofed mammal', 'Domestic canine',
                    'Amphibian', 'Equine mammal', 'Water vessel', 'Large motor vehicle'
                ]
            })
            st.dataframe(class_info, use_container_width=True)
        
        else:
            st.warning("Model not loaded. Please load a model first.")

if __name__ == '__main__':
    main()