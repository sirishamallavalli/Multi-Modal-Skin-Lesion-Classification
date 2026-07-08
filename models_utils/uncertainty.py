import numpy as np

def calculate_mc_dropout_uncertainty(model, image_tensor, meta_tensor, num_passes=20):
    """
    Keeps dropout layers active during test time to run multiple forward 
    passes, computing empirical mean probabilities and stochastic variance boundaries.
    """
    predictions = []
    
    for i in range(num_passes):
        # Forward pass with training=True keeps the Dropout paths active
        preds = model([image_tensor, meta_tensor], training=True)
        predictions.append(preds.numpy())
        
    # Convert predictions to a structured array matrix
    predictions = np.array(predictions) # Shape: (num_passes, batch_size, num_classes)
    
    # Compute the average predictive distribution (Mean)
    mean_probabilities = np.mean(predictions, axis=0)[0]
    
    # Compute predictive variance boundaries (Standard Deviation)
    standard_deviation = np.std(predictions, axis=0)[0]
    
    # Find the predicted class index
    predicted_class = np.argmax(mean_probabilities)
    
    # Isolate variance score belonging to the chosen predicted class
    class_variance = standard_deviation[predicted_class]
    
    return predicted_class, mean_probabilities, class_variance

def check_safety_flag(variance, threshold=0.20):
    """
    Evaluates if internal self-doubt metrics exceed the 
    reliability boundary limits to throw an out-of-distribution alert.
    """
    if variance >= threshold:
        return "WARNING: High Epistemic Uncertainty. Human verification required."
    else:
        return "Stable Prediction Profile."
