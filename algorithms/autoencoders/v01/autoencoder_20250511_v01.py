import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.datasets import mnist
import os
import math

keras.backend.clear_session()

# Define constants
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

# Function to resize an image to the desired dimensions
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

# Function to normalize the pixel values in an image to the range [-1, 1]
def normalize(input_image, real_image):
    input_image = tf.image.rgb_to_grayscale(input_image)
    real_image = tf.image.rgb_to_grayscale(real_image)
    
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

# Function to randomly jitter an image
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)

    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    if tf.random.uniform(()) > 0.5:
        cropped_image = tf.image.flip_left_right(cropped_image)

    return cropped_image[0], cropped_image[1]

# Function to load and preprocess an image
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)    
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# Function to load and preprocess an image for testing (without random jitter)
def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# Load the dataset    
train_dataset = tf.data.Dataset.list_files('./data/TCIRRP/train0.1k/*.jpg')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files('./data/TCIRRP/test0.1k/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

    
class Autoencoder(Model):
    def __init__(self, w, h, latent_dimensions):
        super(Autoencoder, self).__init__()
        
        # Apply convulutional layers and move from flatten to latent hidden layer
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(w, h, 1)), # one channel (intensity)
            layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='relu'),
        ])
        
        self.bw = math.ceil(w / 2**3) # width bottleneck
        self.bh = math.ceil(h / 2**3) # height bottleneck
        self.bc = 128                 # channels bottleneck
                    
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dimensions,)),
            layers.Dense(self.bw * self.bh * self.bc, activation='relu'),
            layers.Reshape((self.bw, self.bh, self.bc)),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1,  3, strides=2, padding='same', activation='relu'),
        ])
    
    
    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Autoencoder(w=256, h=256, latent_dimensions=64)
autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
autoencoder.summary()

print(f"Train dataset cardinality just before fit: {tf.data.experimental.cardinality(train_dataset)}")


autoencoder.fit(
    train_dataset,
    epochs=15,
    validation_data=test_dataset
)

num_tests = 5
x_train = []
x_test = []

for i in range(num_tests):
    input, real = next(iter(train_dataset.take(1))) # since batch size is 1
    x_train.append(input[0])  # Extract the single image from the batch
    x_test.append(real[0])    # Extract the single image from the batch

x_train = tf.stack(x_train)  # Stack the images into a tensor
encoded_imgs = autoencoder.encoder(x_train).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# Plot the original and reconstructed images
plt.figure(figsize=(3, num_tests))


for i in range(num_tests):

    plt.subplot(num_tests, 3, i * 3 + 1)
    plt.imshow(x_train[i].numpy(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(num_tests, 3, i * 3 + 2)
    plt.imshow(decoded_imgs[i], cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.subplot(num_tests, 3, i * 3 + 3)
    plt.imshow(x_test[i].numpy(), cmap='gray')
    plt.title("Real Image")
    plt.axis('off')
    plt.axis('off')
plt.show()

