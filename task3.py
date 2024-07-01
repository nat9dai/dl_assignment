import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Add noise to the dataset
def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

train_images_noisy = add_noise(train_images)
test_images_noisy = add_noise(test_images)

# Create a new model
model_noisy = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model_noisy.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

# Train the model on noisy data
history_noisy = model_noisy.fit(train_images_noisy, train_labels, epochs=10, 
                                validation_data=(test_images_noisy, test_labels))

# Evaluate the model on noisy data
test_loss_noisy, test_acc_noisy = model_noisy.evaluate(test_images_noisy, test_labels, verbose=2)
print(f'Test accuracy with noisy data after re-training: {test_acc_noisy:.4f}')

# Plot training history on noisy data
plt.plot(history_noisy.history['accuracy'], label='accuracy')
plt.plot(history_noisy.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
