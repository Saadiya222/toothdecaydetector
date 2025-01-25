from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the correct path to the model file (go up two levels to the root of 'tooth-decay-detector')
model_path = os.path.join(script_dir, "..", "..", "tooth-decay-detector", "tooth_decay_model.h5")

# Normalize the path
model_path = os.path.normpath(model_path)

#################################################

# Load the model
model = load_model(model_path)

def preprocess_image(img_path):
    """
    Preprocess the image to the required format.
    Args:
    - img_path: Path to the image to be classified.
    
    Returns:
    - preprocessed image ready for prediction.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

def predict_tooth_health(img_path):
    """
    Predicts whether a tooth is healthy or decayed.
    Args:
    - img_path: Path to the tooth image.
    
    Returns:
    - prediction: 'Healthy' or 'Decayed'
    """
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    print(str(prediction))
    return 'Healthy' if prediction < 0.5 else 'Decayed'

# Example usage:

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the image (move up one level to the project root, then into 'data')
image_path = os.path.join(script_dir, "..", "data", "user7.jpg")

# Normalize the path (handles different OS path formats)
image_path = os.path.normpath(image_path)

result = predict_tooth_health(image_path)
print(f'The tooth is {result}.')
