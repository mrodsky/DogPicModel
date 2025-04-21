import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing import image


tf.get_logger().setLevel('ERROR')

# Load the model
model = tf.keras.models.load_model('dog_detector_model_v3.keras')

# Path to the folder with images to check
image_folder = './test_images'

# Function to preprocess and predict if the image has a dog
def is_dog(image_path):
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Rescale
        prediction = model.predict(img_array)
        return prediction[0][0] > 0.5  # Assuming sigmoid activation, threshold at 0.5
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return False

# Iterate over all images in the folder and print the results
for filename in os.listdir(image_folder):
    file_path = os.path.join(image_folder, filename)
    if os.path.isfile(file_path):
        result = is_dog(file_path)
        if result:
            print(f"{filename}: Dog detected")
        else:
            print(f"{filename}: No dog detected")
