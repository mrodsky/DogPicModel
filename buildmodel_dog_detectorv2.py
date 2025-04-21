import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image

# Path to your dataset directory
data_dir = './dog_dataset'

def preprocess_image(image): 
    img = Image.fromarray(np.uint8(image * 255))
    if img.mode == 'P':  # Check if image is in palette mode 
        img = img.convert('RGBA')
        
   
    img_array = np.array(img) 
    return img_array / 255.0 # Rescale
    
# Create an instance of ImageDataGenerator for data preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1./255,
    validation_split=0.4,  # Use 20% of data for validation
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators for training and validation
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='training'  # Use the training subset
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='validation'  # Use the validation subset
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=50
)

# Save the model
model.save('dog_detector_model_v3.keras')
