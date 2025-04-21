import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path to your dataset directory
data_dir = './dog_dataset'

# Create an instance of ImageDataGenerator for data preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create initial data generators to get sample counts
initial_train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

initial_validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Get sample counts
num_train_samples = initial_train_generator.samples
num_validation_samples = initial_validation_generator.samples

# Custom function to convert multi-class labels to binary
def binary_class_mode(labels):
    return np.ones(labels.shape[0])  
# Assuming all images are dogs. For non-dog images, you would set the corresponding labels to 0.

# Custom data generator
def custom_flow_from_directory(generator, directory, subset):
    gen = generator.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset=subset,
        shuffle=True
    )
    while True:
        data_batch, labels_batch = next(gen)
        binary_labels = binary_class_mode(labels_batch)
        yield data_batch, binary_labels

# Create data generators for training and validation
train_generator = custom_flow_from_directory(datagen, data_dir, 'training')
validation_generator = custom_flow_from_directory(datagen, data_dir, 'validation')

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

# Calculate steps per epoch
steps_per_epoch = num_train_samples // 32
validation_steps = num_validation_samples // 32

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2
)


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

plot_history(history)


model.save('dog_detector_model.h5')

