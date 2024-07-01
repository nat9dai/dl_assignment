import tensorflow as tf
from tensorflow.keras import models
import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Add noise to the dataset
def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

test_images_noisy = add_noise(test_images)

# Load the trained model
model = models.load_model('cifar10_model.h5')

# Evaluate the model on noisy data
test_loss_noisy, test_acc_noisy = model.evaluate(test_images_noisy, test_labels, verbose=2)
print(f'Test accuracy with noisy data: {test_acc_noisy:.4f}')
