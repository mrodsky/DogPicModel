from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the model
model = load_model('dog_detector_model_v2.h5')

# Print the model summary
model.summary()

# Plot the model architecture to a file
plot_model(model, to_file='model_architecture_2.png', show_shapes=True, show_layer_names=True)

# Display the model architecture using matplotlib
img = mpimg.imread('model_architecture.png')
imgplot = plt.imshow(img)
plt.show()
