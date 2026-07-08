import cv2
import numpy as np

def remove_hair_and_clean(image_path, radius=5, threshold=10):
    """
    Applies Morphological Blackhat filtering to isolate and 
    inpaint dark hair fibers or pen marks from dermoscopic images.
    """
    # Read image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Structural element for Blackhat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Create mask for hair artifacts
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    
    # Fast Marching Inpainting to fix occluded pixels
    clean_image = cv2.inpaint(img_resized, mask, 1, cv2.INPAINT_TELEA)
    
    # Normalize pixel values between 0 and 1
    clean_image = clean_image.astype('float32') / 255.0
    return clean_image

def process_patient_metadata(age, sex, history_checklist):
    """
    Standardizes and maps patient demographics into a 
    one-hot encoded structure matching the 57-dimensional matrix.
    """
    # Placeholder for the 57-dimensional categorical vector mapping
    meta_vector = np.zeros(57, dtype=np.float32)
    
    # Example logic mapping age and sex features
    meta_vector[0] = float(age) / 100.0  # Simple age scaling
    if sex.lower() == 'male':
        meta_vector[1] = 1.0
    elif sex.lower() == 'female':
        meta_vector[2] = 1.0
        
    # Map checklist items dynamically onto the array matrix
    for idx, checked in enumerate(history_checklist):
        if checked and (idx + 3 < 57):
            meta_vector[idx + 3] = 1.0
            
    return meta_vector
