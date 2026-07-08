import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2
import matplotlib.pyplot as plt
from PIL import Image


st.set_page_config(
    page_title="Multi-Modal Skin Cancer Diagnostic System",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_ml_assets():
    """Loads and caches model, scaler, label encoder, and feature column signatures."""
    model = tf.keras.models.load_model("assets/best_multimodal_mobilenet.keras")
    with open("assets/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("assets/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("assets/meta_columns.pkl", "rb") as f:  
        meta_columns = pickle.load(f)
    return model, scaler, le, meta_columns

try:
    model, scaler, le, meta_columns = load_ml_assets()
    num_classes = len(le.classes_)
except Exception as e:
    st.error(f"⚠️ Error loading assets from 'assets/' folder. Make sure model, scaler, label encoder, and meta_columns are placed correctly. Details: {e}")
    st.stop()

#  HELPER FUNCTIONS: HAIR REMOVAL & MC INFERENCE
def remove_hair(image_np):
    """Applies a morphological filter to remove hair artifacts from normalized inputs."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    clean_img = cv2.inpaint(image_np, thresh, 1, cv2.INPAINT_TELEA)
    return clean_img

def run_mc_dropout_analysis(img_array, meta_vector, iterations=25):
    """Performs stochastic forward passes to compute predictive variance."""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    meta_tensor = tf.convert_to_tensor(meta_vector, dtype=tf.float32)

    preds = []
    for _ in range(iterations):
        out = model([img_tensor, meta_tensor], training=True)
        preds.append(out.numpy()[0])
    
    preds = np.array(preds)
    mean_preds = np.mean(preds, axis=0)
    std_preds = np.std(preds, axis=0)
    return mean_preds, std_preds

# USER INTERFACE LAYOUT
st.title("🔬 Multi-Modal Explainable Skin Cancer Classifier")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("📋 Patient Clinical Inputs")
    
    # Image Input Segment
    uploaded_file = st.file_uploader("Upload Lesion Dermatoscopic Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    # Core Demographic Data
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=45, step=1)
    gender = st.selectbox("Gender", ["MALE", "FEMALE", "unknown"])
    
    st.markdown("### 🔍 Affected Area Symptom Profiles")
    # Top 5-6 Questions targeting the lesion/affected area behavior specifically
    has_itch = st.selectbox("Is the affected area itchy (Itching)?", ["False", "True", "unknown"])
    has_grew = st.selectbox("Has the lesion grown in size recently (Growth)?", ["False", "True", "unknown"])
    has_bleed = st.selectbox("Does the affected area bleed?", ["False", "True", "unknown"])
    has_hurt = st.selectbox("Is the lesion painful or tender (Hurt)?", ["False", "True", "unknown"])
    has_elevation = st.selectbox("Is the lesion raised/elevated profile?", ["False", "True", "unknown"])
    
    st.markdown("### 🚬 Personal Clinical History")
    has_history = st.selectbox("Personal/Family History of Skin Cancer?", ["False", "True", "unknown"])
    is_smoker = st.selectbox("Does the patient smoke tobacco?", ["False", "True", "unknown"])
    
    apply_preprocessing = st.checkbox("Activate Digital Hair Removal Filter", value=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Run Diagnostic Evaluation", type="primary", use_container_width=True)

# DATA PROCESSING AND INFERENCE PIPELINE
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_image)
    
    with col1:
        st.image(pil_image, caption="Uploaded Documented Lesion", use_container_width=True)

    if analyze_btn:
        with col2:
            st.header("📊 Clinical Diagnostic Report")
            
            with st.spinner("Processing image textures and encoding clinical metadata..."):
                # A. Image Preprocessing 
                if apply_preprocessing:
                    processed_np = remove_hair(original_np)
                else:
                    processed_np = original_np.copy()
                
                img_resized = cv2.resize(processed_np, (128, 128))
                img_normalized = img_resized / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)

                # B. Metadata Alignment Pipeline 
                # Create a baseline row filled entirely with zeros based on training layout
                input_row = {col: 0.0 for col in meta_columns}

                # 1. Assign the numeric age safely
                if 'age' in input_row:
                    input_row['age'] = float(age)

                # 2. Handle Gender mapping dynamically 
                for col in input_row.keys():
                    if col.startswith(('gender_', 'sex_')):
                        if gender.lower() in col.lower():
                            input_row[col] = 1.0

                # 3. Handle Affected Area characteristics dynamically
                ui_features = {
                    'itch': has_itch,
                    'grew': has_grew,
                    'bleed': has_bleed,
                    'hurt': has_hurt,
                    'elevation': has_elevation,
                    'skin_cancer_history': has_history,
                    'smoke': is_smoker
                }

                # Map selection parameters into their corresponding multi-column dummy arrays
                for feature_name, selection in ui_features.items():
                    for col in input_row.keys():
                        if col.startswith(f"{feature_name}_"):
                            # If column matches selection string value (e.g., itch_True or itch_unknown)
                            if selection.lower() in col.lower():
                                input_row[col] = 1.0

                # 4. Extract values in the exact structural sequence required by the scaler matrix
                meta_vector = np.array([[input_row[col] for col in meta_columns]])
                meta_vector = np.nan_to_num(meta_vector.astype(np.float32))

                # Scale the metadata cleanly
                meta_scaled = scaler.transform(meta_vector)

                # C. Run MC Dropout Inference 
                mean_preds, std_preds = run_mc_dropout_analysis(img_batch, meta_scaled, iterations=50)
                
                pred_class_idx = np.argmax(mean_preds)
                pred_class_name = le.classes_[pred_class_idx]
                confidence_val = mean_preds[pred_class_idx]
                uncertainty_val = std_preds[pred_class_idx]
                
                # D. Render Metrics Dashboard 
                safety_threshold = 0.20
                
                st.subheader(f"Primary Indication: **{pred_class_name}**")
                
                m1, m2 = st.columns(2)
                m1.metric("Mean Prediction Confidence", f"{confidence_val * 100:.1f}%")
                m2.metric("Stochastic Uncertainty (σ)", f"{uncertainty_val:.4f}")
                
                if uncertainty_val > safety_threshold:
                    st.error("⚠️ **ALERT: HIGH STOCHASTIC UNCERTAINTY DETECTED**\n\nThe prediction patterns fluctuate aggressively when internal neurons are dropped out. Diagnostic reliability threshold breached. **Recommendation: REFER TO SPECIALIST for physical biopsy.**")
                else:
                    st.success("✅ **Status: Confirmed Diagnostics**\n\nStochastic variances remain within normal ranges. Standard clinical charting protocols advised.")

                st.markdown("---")
                
                # E. Render Analytical Charts 
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
                classes = le.classes_
                x_pos = np.arange(len(classes))
                
                # Left: Prediction Confidence
                bars1 = ax1.bar(x_pos, mean_preds, color='#8cd975', edgecolor='black')
                ax1.set_title("Mean Prediction Probabilities", fontsize=11, fontweight='bold')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(classes, fontsize=9)
                ax1.set_ylim(0, 1.1)
                for bar in bars1:
                    yval = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval*100:.1f}%", ha='center', va='bottom', fontsize=8)
                
                # Right: Stochastic Uncertainty Chart
                bars2 = ax2.bar(x_pos, std_preds, color='salmon', alpha=0.8, edgecolor='black')
                ax2.set_title("Stochastic Variances (Uncertainty)", fontsize=11, fontweight='bold')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(classes, fontsize=9)
                ax2.axhline(y=safety_threshold, color='r', linestyle='--', label=f'Safety Limit ({safety_threshold})')
                ax2.set_ylim(0, max(0.35, np.max(std_preds) + 0.05))
                ax2.legend(loc='upper right', prop={'size': 8})
                
                plt.tight_layout()
                st.pyplot(fig)
else:
    with col2:
        st.info("Waiting for image drop and clinical parameter submission from left panel to generate diagnostic insights...")
